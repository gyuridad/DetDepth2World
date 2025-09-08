import cv2
import numpy as np
import torch

import depth_pro


if __name__=="__main__":
    input_img_path = "data/frame_00000.jpg"
    output_depth_path = "results/depth.npy"
    output_depth_vis_path = "results/depth_vis.jpg"
    output_ply_path = "results/point_cloud.ply"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.float16)
    depth_model.eval()

    input_img, _, focal_px = depth_pro.load_rgb(input_img_path)
    input_img = transform(input_img)

    # 초점거리가 None이면 별도의 초점거리 추정 헤드가 이미지 특징으로부터 **가로 시야각(FOV_x)**을 예측합니다.
    # 이 헤드는 깊이 네트워크와 분리해 따로 학습되며, FOV를 잘 맞추도록 설계돼 있어요.
    result = depth_model.infer(input_img, f_px=focal_px)

    # result["depth"]는 **메트릭 깊이(카메라로부터의 거리)**를 미터 [m] 단위로 반환
    output_depth = result["depth"].cpu().numpy()
    output_depth = output_depth * 1000.0    # 1000.0을 곱해 **밀리미터 [mm]**로 바꿈 
    ### 왜 mm로 바꾸냐면:
    # 16비트 깊이 이미지 호환: PNG/TIFF로 저장할 때는 정수형이 유리해서, 보통 mm 단위의 uint16(0~65535)을 씁니다. (1mm 정밀도)
    # 툴/센서와의 관례: RealSense/Kinect, ROS/OpenCV/Open3D 예제들이 mm(또는 depth_scale=1000) 관례를 많이 써요.
    # 가시화 편의: 값 범위가 커져 히스토그램/시각화가 직관적일 때가 있습니다.

    output_focal_px = result["focallength_px"].cpu().numpy()
    ### 초점거리(px) = 이미지 좌표(픽셀)로 계산하려고, **광학 초점거리(mm)**를 픽셀 크기로 나눠서 픽셀 단위로 바꿔놓은 값이에요.
    # 즉, 이미지 평면에서 “얼마나 멀리”가 몇 픽셀인지 나타내는 스케일입니다.

    depth_vis_img = np.uint8(cv2.normalize(output_depth, None, 0, 255, cv2.NORM_MINMAX))
    # norm_type=cv2.NORM_MINMAX → 선형 스케일링 사용
    # 그다음 np.uint8(...)로 8비트 정수로 캐스팅 → 그레이스케일 이미지로 바로 저장/표시 가능

    np.save(output_depth_path, output_depth)
    cv2.imwrite(output_depth_vis_path, depth_vis_img)

    # print(output_depth)
    # print(output_focal_px)