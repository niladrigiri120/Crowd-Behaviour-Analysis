import os
import cv2
import torch
from torch.utils.data import Dataset

class OpticalFlowSequenceDataset(Dataset):
    def __init__(self, flow_dir, sequence_length=10, stride=3):
        self.flow_dir = flow_dir
        self.sequence_length = sequence_length
        self.stride = stride

        self.frames = sorted(os.listdir(flow_dir))

    def __len__(self):
        return (len(self.frames) - self.sequence_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        seq = []

        for i in range(self.sequence_length):
            img_path = os.path.join(
                self.flow_dir, self.frames[start + i]
            )
            img = cv2.imread(img_path)

            if img is None:
                raise RuntimeError(f"Failed to read {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
            seq.append(img)

        seq = torch.stack(seq)  # T,C,H,W
        return seq
