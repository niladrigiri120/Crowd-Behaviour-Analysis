import streamlit as st
import requests
import matplotlib.pyplot as plt

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Crowd Behavior Analysis",
    layout="wide"
)

st.title("🚨 Real-Time Multi-Camera Crowd Behavior Analysis")

st.subheader("📹 Live Camera Feed")

st.divider()
st.subheader("📹 Live Multi-Camera Feeds")

CAMERAS = ["cam1", "cam2", "cam3", "cam4"]

cols = st.columns(2)  # 2x2 grid

for idx, cam in enumerate(CAMERAS):
    with cols[idx % 2]:
        st.markdown(f"**{cam.upper()}**")
        st.image(
            f"http://127.0.0.1:8000/video/{cam}",
            width="stretch"
        )


# -----------------------------
# Fetch backend data
# -----------------------------
@st.cache_data(ttl=5)
def fetch_status():
    return requests.get(f"{API_BASE}/status").json()

@st.cache_data(ttl=5)
def fetch_anomalies():
    return requests.get(f"{API_BASE}/anomalies").json()

status_data = fetch_status()
anomaly_data = fetch_anomalies()

camera_status = status_data["camera_status"]
global_alert = status_data["global_alert"]

# -----------------------------
# Global Alert Banner
# -----------------------------
if global_alert:
    st.error("🚨 GLOBAL CROWD ANOMALY DETECTED")
else:
    st.success("✅ Crowd Behavior Normal")

st.divider()

# -----------------------------
# Camera-wise Status
# -----------------------------
st.subheader("📹 Camera Status")

cols = st.columns(len(camera_status))

for col, (cam, state) in zip(cols, camera_status.items()):
    with col:
        if state == "ABNORMAL":
            st.error(f"{cam}\nABNORMAL")
        else:
            st.success(f"{cam}\nNORMAL")

st.divider()

# -----------------------------
# Anomaly Timeline Visualization
# -----------------------------
st.subheader("📈 Anomaly Timeline")

camera_anomalies = anomaly_data["camera_anomalies"]

for cam, times in camera_anomalies.items():
    if len(times) == 0:
        continue

    st.markdown(f"### {cam}")

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.scatter(times, [1] * len(times), color="red")
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"{cam} – Detected Anomalies")
    st.pyplot(fig)

st.divider()

# -----------------------------
# Explainability Section
# -----------------------------
st.subheader("🧠 Explainability")

st.markdown("""
- The model learns **normal crowd motion patterns** using optical flow.
- **Reconstruction error** is used as an anomaly score.
- Peaks in the timeline indicate **unusual crowd behavior**.
- Detection is based on **group-level motion**, not individuals.
""")

st.divider()

# -----------------------------
# Raw Data Viewer (Optional)
# -----------------------------
with st.expander("🔍 View Raw Anomaly Data"):
    st.json(anomaly_data)
