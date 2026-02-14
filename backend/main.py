from fastapi import FastAPI
import json
import os
import cv2
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="Crowd Behavior Analysis API",
    description="Multi-camera crowd anomaly detection backend",
    version="1.0"
)

RESULT_FILE = "results/all_camera_anomalies.json"

def load_results():
    if not os.path.exists(RESULT_FILE):
        return {}
    with open(RESULT_FILE) as f:
        return json.load(f)

@app.get("/")
def root():
    return {"status": "Crowd Behavior Analysis API running"}

@app.get("/status")
def get_status():
    data = load_results()
    camera_data = data.get("camera_anomalies", {})

    status = {}
    for cam, times in camera_data.items():
        status[cam] = "ABNORMAL" if len(times) > 0 else "NORMAL"

    global_alert = any(len(v) > 0 for v in camera_data.values())

    return {
        "camera_status": status,
        "global_alert": global_alert
    }

@app.get("/anomalies")
def get_anomalies():
    data = load_results()
    return data


# ----------------------------------
# Define camera sources
# ----------------------------------
CAMERA_SOURCES = {
    "cam1": "data/videos/cam1.avi",
    "cam2": "data/videos/cam2.avi",
    "cam3": "data/videos/cam3.avi",
    "cam4": "data/videos/cam4.avi",
}

def generate_frames(source):
    cap = cv2.VideoCapture(source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.get("/video/{camera_id}")
def video_feed(camera_id: str):
    if camera_id not in CAMERA_SOURCES:
        return {"error": "Invalid camera ID"}

    return StreamingResponse(
        generate_frames(CAMERA_SOURCES[camera_id]),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
