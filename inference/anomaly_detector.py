import sys
import os
import cv2
import torch
import numpy as np

# -----------------------------
# Fix imports (project root)
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from models.convlstm import ConvLSTMAutoencoder

# -----------------------------
# Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/convlstm_autoencoder.pth"

# IMPORTANT: use LOCAL disk path in Colab
FLOW_PATH = "data/flow/cam1"   # change cam if needed

SEQUENCE_LENGTH = 10
STRIDE = 10           # BIG speed-up
BATCH_SIZE = 4        # safe for Colab GPU
MAX_FRAMES = 1500      # limit for demo / validation

# -----------------------------
# Load model
# -----------------------------
model = ConvLSTMAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Load flow images (ONCE)
# -----------------------------
flow_files = sorted(os.listdir(FLOW_PATH))[:MAX_FRAMES]

print(f"Using {len(flow_files)} flow frames")

sequences = []

for i in range(0, len(flow_files) - SEQUENCE_LENGTH, STRIDE):
    seq = []
    for j in range(SEQUENCE_LENGTH):
        img_path = os.path.join(FLOW_PATH, flow_files[i + j])
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        seq.append(img)

    if len(seq) == SEQUENCE_LENGTH:
        sequences.append(torch.stack(seq))

# Stack all sequences
sequences = torch.stack(sequences).to(DEVICE)
print(f"Total sequences: {sequences.shape[0]}")

# -----------------------------
# Batched inference
# -----------------------------
errors = []

with torch.no_grad():
    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i:i+BATCH_SIZE]
        recon = model(batch)
        batch_error = torch.mean((recon - batch) ** 2, dim=(1,2,3,4))
        errors.extend(batch_error.cpu().numpy())

errors = np.array(errors)

print("Inference completed")
print("Mean error:", errors.mean())
print("Std error :", errors.std())

threshold = errors.mean() + 3 * errors.std()

labels = errors > threshold

print("Threshold:", threshold)
print("Abnormal events detected:", labels.sum())

# -----------------------------
# Visualization: Anomaly Timeline
# -----------------------------
import matplotlib.pyplot as plt

abnormal_indices = [i for i, v in enumerate(errors) if v > threshold]

# After computing abnormal_indices
FPS = 10
STRIDE = 10

abnormal_times = [i * STRIDE / FPS for i in abnormal_indices]

plt.figure(figsize=(12, 4))
plt.plot(errors, label="Anomaly Score", linewidth=2)
plt.axhline(threshold, color="red", linestyle="--", label="Threshold")

plt.scatter(
    abnormal_indices,
    [errors[i] for i in abnormal_indices],
    color="red",
    s=60,
    label="Detected Anomaly"
)

plt.title("Crowd Behavior Anomaly Detection Timeline")
plt.xlabel("Time Window Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
