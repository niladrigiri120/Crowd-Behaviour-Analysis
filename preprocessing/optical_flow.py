import cv2
import os
import numpy as np

FRAME_PATH = os.path.abspath("data/processed")
FLOW_PATH = os.path.abspath("data/flow")

CAMERAS = ["cam1", "cam2", "cam3", "cam4"]

os.makedirs(FLOW_PATH, exist_ok=True)

for cam in CAMERAS:
    cam_frame_dir = os.path.join(FRAME_PATH, cam)
    cam_flow_dir = os.path.join(FLOW_PATH, cam)
    os.makedirs(cam_flow_dir, exist_ok=True)

    frames = sorted(os.listdir(cam_frame_dir))
    print(f"\nProcessing optical flow for {cam}")

    prev_frame = cv2.imread(os.path.join(cam_frame_dir, frames[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_frame = cv2.imread(os.path.join(cam_frame_dir, frames[i]))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(curr_frame)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        flow_name = f"flow_{i:06d}.jpg"
        cv2.imwrite(os.path.join(cam_flow_dir, flow_name), flow_image)

        prev_gray = curr_gray

    print(f"Optical flow saved for {cam}")
