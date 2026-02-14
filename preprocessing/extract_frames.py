import cv2
import os

# Paths
VIDEO_PATH = os.path.abspath("data/videos")
OUTPUT_PATH = os.path.abspath("data/processed")

VIDEOS = ["cam1.avi", "cam2.avi", "cam3.avi", "cam4.avi"]

TARGET_FPS = 10
RESIZE_TO = (224, 224)

os.makedirs(OUTPUT_PATH, exist_ok=True)

for video in VIDEOS:
    cam_name = video.split(".")[0]
    video_file = os.path.join(VIDEO_PATH, video)
    output_dir = os.path.join(OUTPUT_PATH, cam_name)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(round(original_fps / TARGET_FPS))
    frame_count = 0
    saved_count = 0

    print(f"\nProcessing {video}")
    print(f"Original FPS: {original_fps}, Saving every {frame_interval} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, RESIZE_TO)
            frame_name = f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames for {cam_name}")
