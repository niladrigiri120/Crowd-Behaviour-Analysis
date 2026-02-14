import cv2
import os
import time

BASE_PATH = os.path.abspath("data/videos")
VIDEO_FILES = ["cam1.avi", "cam2.avi", "cam3.avi", "cam4.avi"]

TARGET_FPS = 10                 # simulate real-time (10 FPS)
FRAME_DELAY = 1 / TARGET_FPS
RESIZE_TO = (224, 224)          # standard CNN size

caps = []

# Open all camera streams
for video in VIDEO_FILES:
    path = os.path.join(BASE_PATH, video)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video}")
    else:
        print(f"[OK] Opened {video}")

    caps.append(cap)

print("\nStarting real-time multi-camera stream...\n")

while True:
    frames = []

    for idx, cap in enumerate(caps):
        ret, frame = cap.read()

        if not ret:
            print(f"[INFO] Stream ended for cam{idx+1}")
            cap.release()
            continue

        frame = cv2.resize(frame, RESIZE_TO)
        cv2.putText(
            frame,
            f"Camera {idx+1}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        frames.append(frame)

    # Display each camera in its own window
    for i, frame in enumerate(frames):
        cv2.imshow(f"Cam {i+1}", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(FRAME_DELAY)

for cap in caps:
    cap.release()

cv2.destroyAllWindows()
