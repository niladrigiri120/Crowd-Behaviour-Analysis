import cv2
import os

BASE_PATH = os.path.abspath("data/videos")

videos = ["cam1.avi", "cam2.avi", "cam3.avi", "cam4.avi"]

print("Base path:", BASE_PATH)

for video in videos:
    video_path = os.path.join(BASE_PATH, video)
    print("\nTrying:", video_path)

    if not os.path.exists(video_path):
        print(f"[MISSING] File not found: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0

    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Duration   : {duration/60:.2f} minutes")

    cap.release()
