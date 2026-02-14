import numpy as np

# -----------------------------
# Example anomaly times per camera (seconds)
# Replace these with real outputs
# -----------------------------
camera_anomalies = {
    "cam1": [32.0, 78.0],
    "cam2": [33.5],
    "cam3": [],
    "cam4": [79.2]
}

TIME_WINDOW = 2.0   # seconds
MIN_CAMERAS = 2     # voting threshold

# -----------------------------
# Aggregate anomalies
# -----------------------------
all_events = []

for cam, times in camera_anomalies.items():
    for t in times:
        all_events.append((t, cam))

all_events.sort(key=lambda x: x[0])

global_alerts = []

for i, (time_i, cam_i) in enumerate(all_events):
    supporting_cams = {cam_i}

    for j, (time_j, cam_j) in enumerate(all_events):
        if i != j and abs(time_i - time_j) <= TIME_WINDOW:
            supporting_cams.add(cam_j)

    if len(supporting_cams) >= MIN_CAMERAS:
        global_alerts.append({
            "time": time_i,
            "cameras": list(supporting_cams)
        })

# Remove duplicates
unique_alerts = []
seen_times = set()

for alert in global_alerts:
    t = round(alert["time"])
    if t not in seen_times:
        unique_alerts.append(alert)
        seen_times.add(t)

# -----------------------------
# Output
# -----------------------------
print("GLOBAL CROWD ALERTS")
for alert in unique_alerts:
    print(
        f"Time: {alert['time']:.1f}s | Cameras: {alert['cameras']}"
    )
