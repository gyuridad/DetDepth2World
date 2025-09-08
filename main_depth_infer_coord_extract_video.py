import os
import cv2
import numpy as np
import math
import torch
from PIL import Image

from ultralytics import YOLO 
import depth_pro

# ---------- 유틸 함수들 ----------
def robust_depth_in_roi(depth_m, bbox=None, mask=None):
    """
    depth_m: (H,W) float32 미터[m]
    bbox: (x1,y1,x2,y2)
    mask: (H,W) bool
    return: Z_med (m), stats(dict)
    """
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        H, W = depth_m.shape
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            return None, {"valid": 0}
        roi = depth_m[y1:y2, x1:x2]
    elif mask is not None:
        roi = depth_m[mask]
    else:
        raise ValueError("bbox나 mask 중 하나는 필요합니다.")

    roi = roi[np.isfinite(roi)]
    roi = roi[(roi > 0) & (roi < 1e4)]  # 말도 안 되게 큰 값 컷(필요시 조정)

    if roi.size == 0:
        return None, {"valid": 0}

    lo, hi = np.percentile(roi, [10, 90])  # 이상치 제거
    roi_c = roi[(roi >= lo) & (roi <= hi)]
    if roi_c.size == 0:
        roi_c = roi

    Z_med = float(np.median(roi_c))
    stats = {"valid": int(roi.size), "p10": float(lo), "p90": float(hi), "median": Z_med}
    return Z_med, stats


def pixel_to_cam(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return float(X), float(Y), float(Z)

def object_3d_from_bbox(depth_m, bbox, fx, fy, cx, cy):
    # 1) ROI에서 대표 깊이
    Z, stats = robust_depth_in_roi(depth_m, bbox=bbox)
    if Z is None:
        return None, stats

    # 2) bbox 중심 픽셀
    x1, y1, x2, y2 = map(int, bbox)
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    # 3) 픽셀 -> 카메라 좌표
    X, Y, Z = pixel_to_cam(u, v, Z, fx, fy, cx, cy)
    dist = float(np.sqrt(X * X + Y * Y + Z * Z))

    return {"X": X, "Y": Y, "Z": Z, "dist": dist, "u": u, "v": v}, stats

# def ground_range_from_dist(dist, cam_height, obj_height=0.0):
#     dh = float(cam_height - obj_height)
#     base = dist*dist - dh*dh
#     if base <= 0 or not np.isfinite(base):
#         return None, {"ok": False, "dist": dist, "dh": dh, "base": base}
#     return math.sqrt(base), {"ok": True, "dist": dist, "dh": dh, "base": base}

#### obj_height는 0.0 으로 해도 돼?
# 네, 대상이 ‘지면 위에 놓여 있다’고 보면 obj_height = 0.0으로 써도 됩니다.
# 언제 0.0이 합리적?
        # 사람/차량/콘/박스처럼 도로·지면에 붙어 있는 물체
# 0.0이 아니어야 할 때는?
        # 옥상/다리 위/계단/단차 등 지면보다 높이 있는 대상
        # 표지판/전봇대 상단/드론 등 공중/고지대 대상
# 실무 팁:
        # 클래스별 근사 높이 넣으면 더 안정적:
        # DEFAULT_OBJ_H = {'person': 1.7, 'car': 1.5, 'truck': 2.5, 'bus': 3.2}
        # obj_h = DEFAULT_OBJ_H.get(label, 0.0)  # label은 YOLO 클래스명

# ---------- 카메라좌표를 월드 좌표로 변환 관련 함수들 ----------

def rot_zyx(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """
    R_wc = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    - world <- camera  회전행렬 (카메라벡터를 월드좌표로 회전)
    - 각도 단위: degree
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cz, sz = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cx, sx = np.cos(r), np.sin(r)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=float)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=float)
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=float)

    return Rz @ Ry @ Rx

def get_R_bc(mount="nadir"):
    """
    Camera(OpenCV: +x=right, +y=down, +z=forward) -> Body(FLU: +X=forward, +Y=left, +Z=up)
    """
    # 전방(정면) 장착
    R_bc_front = np.array([
        [ 0,  0,  1],   # cZ -> bX
        [-1,  0,  0],   # cX -> -bY
        [ 0, -1,  0],   # cY -> -bZ
    ], float)

    # 하향(나디르) 장착
    R_bc_nadir = np.array([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0, -1],
    ], float)

    if mount == "front":
        return R_bc_front
    elif mount == "nadir":
        return R_bc_nadir
    else:
        raise ValueError("mount must be 'front' or 'nadir'")

def cam_to_world_point(X, Y, Z, yaw_deg, pitch_deg, roll_deg, Cw, mount="nadir", R_bc=None):
    """
    카메라 좌표 P_c=[X,Y,Z] (OpenCV: x=오른, y=아래, z=앞)를
    월드(ENU) 좌표로 변환: P_w = (R_wb @ R_bc) @ P_c + C_w
    - mount: 'nadir'(기본) 또는 'front'
    - R_bc: 직접 보정행렬을 넣고 싶으면 전달(우선순위 높음)
    """
    R_wb = rot_zyx(yaw_deg, pitch_deg, roll_deg)        # Body -> World
    R_bc = R_bc if R_bc is not None else get_R_bc(mount) # Camera -> Body
    R_wc = R_wb @ R_bc                                  # Camera -> World

    Pc = np.array([X, Y, Z], dtype=float)
    Cw = np.array(Cw, dtype=float)
    Pw = R_wc @ Pc + Cw
    return Pw


# ---------- 메인 파이프라인 ----------
if __name__=="__main__":
    # input_img_path = "data/frame_00005.jpg"
    output_npy_dir = "results/npy_frames"
    output_dir = "results/depth_frames"
    output_overlay = "results/yolo_frames"
    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture("data/지노드론내서읍.mp4")

    ### 아래 파라미터는 실제 파라미터로 바꿔야 함 !!! ###
    Cw = [100.0, 100.0, 0.0]     # 드론의 월드 좌표 (E, N, U) [m]
    yaw_deg = 0.0                # 방위(0°=East, +CCW)
    pitch_deg = -30.0             # 하향이면 -, 상향이면 +
    roll_deg = 0.0
    # ----------------------------------

    # YOLO 가중치 (원하면 커스텀 pt로 바꿔도 됨)
    yolo_weights = "yolov11m_myeong_best.pt" # yolov8n / yolo11m / yolov11m_myeong_best
    yolo_conf = 0.4
    class_filter = None    # 예: [0]만(=person) 보려면 [0]로 설정

    # ---- 성능 옵션 ----
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(yolo_weights)
    model = yolo.model

    # 1) 반드시 float32에서 fuse
    model.float()
    model = model.fuse(verbose=False)   # Conv+BN 결합 (float32 상태에서)

    # 2) 디바이스 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3) GPU라면 half로 변환
    if device == "cuda":
        model.half()

    # 4) 다시 모델을 YOLO 래퍼에 반영
    yolo.model = model

    # DepthPro 모델/변환 준비
    depth_model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    depth_model.eval()

    focal_px = None
    frame_idx = 0
    while video.isOpened():
        ret, bgr = video.read()
        if not ret:
            break

        yolo_res = yolo(bgr, conf=yolo_conf)[0]

        H0, W0 = bgr.shape[:2]
        # print(f"H0, W0 => {H0}, {W0}")

        # 3) DepthPro 입력 로드 (focal_px 얻기)
        #    load_rgb가 PIL.Image를 반환한다고 가정 (모듈에 맞춰 사용)
        # rgb_pil, _, focal_px = depth_pro.load_rgb(input_img_path)  # focal_px는 픽셀 단위   ---> 아래 코드 사용 !

        run_depth_now = (frame_idx % 2 == 0)

        # OpenCV(BGR) → PIL(RGB)
        if run_depth_now:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_pil = Image.fromarray(rgb)

            rgb_tensor = transform(rgb_pil)  # 네트워크 입력 텐서
            rgb_tensor = rgb_tensor.to(device, dtype=torch.float16, non_blocking=True)

            # 초점거리가 None이면 별도의 초점거리 추정 헤드가 이미지 특징으로부터 **가로 시야각(FOV_x)**을 예측합니다.
            # 이 헤드는 깊이 네트워크와 분리해 따로 학습되며, FOV를 잘 맞추도록 설계돼 있어요.
            with torch.no_grad():
                result = depth_model.infer(rgb_tensor, f_px=focal_px)

            # result["depth"]는 **메트릭 깊이(카메라로부터의 거리)**를 미터 [m] 단위로 반환
            # detach()는 텐서를 연산 그래프에서 분리(detach) 한 뒤 CPU로 옮겨 NumPy 뷰를 만드는 안전한 권장 방식
            depth_m_net = result["depth"].detach().cpu().numpy()     # (Hd, Wd) 미터

            # focal_px_net = result["focallength_px"].detach().cpu().numpy()
            ### 초점거리(px) = 이미지 좌표(픽셀)로 계산하려고, **광학 초점거리(mm)**를 픽셀 크기로 나눠서 픽셀 단위로 바꿔놓은 값이에요.
            # 즉, 이미지 평면에서 “얼마나 멀리”가 몇 픽셀인지 나타내는 스케일입니다.

            # (1) focal(px) 스칼라 안전 추출
            f_any = result["focallength_px"]
            if torch.is_tensor(f_any):
                focal_px_net = float(f_any.detach().cpu().item())
            elif isinstance(f_any, np.ndarray):
                focal_px_net = float(f_any.reshape(-1)[0])
            else:
                focal_px_net = float(f_any)  # 숫자면 캐스팅

            # (2) 해상도 정보
            Hd, Wd = depth_m_net.shape[-2:]
            # print(f"Hd, Wd => {Hd}, {Wd}")

            # (3) intrinsics를 "깊이맵 해상도(Wd,Hd) 기준"으로 먼저 정의
            fx0 = fy0 = focal_px_net
            cx0, cy0 = Wd / 2.0, Hd / 2.0

            # cv2.resize(...)를 둔 이유는 **YOLO bbox 좌표계(원본 이미지 기준)**와 깊이맵 좌표계가 다를 수 있는 상황에 대비한 “안전장치”
            if (Hd, Wd) != (H0, W0):
                depth_m = cv2.resize(depth_m_net, (W0, H0), interpolation=cv2.INTER_NEAREST)
                # intrinsics도 같이 스케일
                sx, sy = W0 / float(Wd), H0 / float(Hd)
                fx, fy = fx0 * sx, fy0 * sy
                cx, cy = cx0 * sx, cy0 * sy
            else:
                depth_m = depth_m_net
                fx, fy = fx0, fy0
                cx, cy = cx0, cy0
                # 스케일 = 1 → fx,fy,cx,cy 그대로 사용

            print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")

            # 깊이 시각화 저장(선택)
            depth_mm = depth_m * 1000.0
            depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"depth_vis_{frame_idx:06d}.jpg"), depth_vis)
            np.save(os.path.join(output_npy_dir, f"depth_{frame_idx:06d}.npy"), depth_m)  # 미터 단위 저장
            cached_depth = depth_m
            cached_K = (fx, fy, cx, cy)

        overlay = bgr.copy()
        names = yolo_res.names
        detections = yolo_res.boxes

        if detections is None or detections.data.shape[0] == 0:
            print("YOLO: 검출 없음")
        else:
            for i in range(detections.data.shape[0]):
                # 박스/점수/클래스
                xyxy = detections.xyxy[i].cpu().numpy().tolist()   # [x1,y1,x2,y2]
                conf = float(detections.conf[i].cpu().numpy())
                cls_id = int(detections.cls[i].cpu().numpy())
                if class_filter is not None and cls_id not in class_filter:
                    continue

                ### 깊이에서 3D/거리 추정
                obj3d, stats = object_3d_from_bbox(depth_m, xyxy, fx, fy, cx, cy)

                # 디버그 출력
                if obj3d is None:
                    print(f"[{i}] {names.get(cls_id, cls_id)} conf={conf:.2f} | 유효 깊이 없음 (valid={stats.get('valid',0)})")
                else:
                    X = obj3d.get("X")
                    Y = obj3d.get("Y")
                    Z = obj3d.get("Z")
                    R = obj3d.get("dist")         # 카메라-객체 직선거리
                
                # 월드좌표 변환
                Pw = cam_to_world_point(X, Y, Z, yaw_deg, pitch_deg, roll_deg, Cw, mount="front", R_bc=None)

                # 그리기
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"

                def fmt(v): return f"{v:.2f}" if (v is not None and np.isfinite(v)) else "N/A"

                label += f" | D={fmt(obj3d['dist'])}m | W_Coord=[{Pw[0]:.2f},{Pw[1]:.2f},{Pw[2]:.2f}]"
                ### "dist" 예를들면, X=0.21, Y=0.05, Z=2.39 (m) 일때 → 카메라에서 오른쪽 0.21 m, 아래 0.05 m, 앞으로 2.39 m 지점에 있는 3D 위치

                # 라벨 배경
                (tw, th), baseline = cv2.getTextSize(
                    text=label,                 # 문자열
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴(허시 폰트)
                    fontScale=0.5,              # 글자 크기 배율
                    thickness=1                 # 획 두께(픽셀)
                )
                ### 반환값:
                    # tw (text width): 문자열의 가로폭(픽셀)
                    # th (text height): 기본선(baseline) 위쪽에 있는 글자의 세로높이(픽셀)
                    # baseline: 글자를 그릴 때 기본선 아래쪽으로 필요한 공간(픽셀)

                cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # 중심점 찍기(옵션)
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                cv2.circle(overlay, (u, v), 3, (0, 0, 255), -1)

                print(
                    f"[{i}] {names.get(cls_id, cls_id)} conf={conf:.2f} | "
                    f"X={fmt(X)} Y={fmt(Y)} Z={fmt(Z)} m | R={fmt(R)} m | "
                    f"u={fmt(u)} v={fmt(v)} "
                )
                print(f"월드좌표={Pw}")

        cv2.imwrite(os.path.join(output_overlay, f"depth_vis_{frame_idx:06d}.jpg"), overlay)
        # print(f"✔ 저장: {output_overlay}")
        # print(f"✔ 저장: {output_depth_vis}")
        # print(f"✔ 저장: {output_depth_npy}")


        # 화면 출력 전에 프레임 크기 줄이기
        resized_frame = cv2.resize(bgr, (1280, 720))  # 또는 fx=0.5, fy=0.5

        # 화면 출력
        cv2.imshow("Depth Infer Coord Extract", resized_frame)

        # ESC 키 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    video.release()
    cv2.destroyAllWindows()
