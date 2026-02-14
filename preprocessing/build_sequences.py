import os
import cv2
import numpy as np

FLOW_PATH = os.path.abspath("data/flow")
SEQ_PATH = os.path.abspath("data/sequences")

CAMERAS = ["cam1", "cam2", "cam3", "cam4"]
SEQUENCE_LENGTH = 10
STRIDE = 3   # speed-up

os.makedirs(SEQ_PATH, exist_ok=True)

for cam in CAMERAS:
    cam_flow_dir = os.path.join(FLOW_PATH, cam)
    flow_files = sorted(os.listdir(cam_flow_dir))

    print(f"\nBuilding sequences for {cam}")

    sequences = []
    skipped = 0

    for i in range(0, len(flow_files) - SEQUENCE_LENGTH, STRIDE):
        seq = []
        valid = True

        for j in range(SEQUENCE_LENGTH):
            img_path = os.path.join(cam_flow_dir, flow_files[i + j])
            img = cv2.imread(img_path)

            if img is None:
                valid = False
                skipped += 1
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            seq.append(img)

        if valid:
            sequences.append(seq)

    sequences = np.array(sequences, dtype=np.float32)

    out_file = os.path.join(SEQ_PATH, f"{cam}_sequences.npy")
    np.save(out_file, sequences)

    print(f"Saved {sequences.shape[0]} sequences for {cam}")
    print(f"Skipped {skipped} sequences for {cam}")
