import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HiRIDLOSDataset(Dataset):
    def __init__(self, npy_dir, label_path, max_len=256):
        self.data = []
        self.labels = []

        # Load label CSV
        label_df = pd.read_csv(label_path)
        label_map = dict(zip(label_df["patientid"], label_df["los_days"]))

        for filename in os.listdir(npy_dir):
            if not filename.endswith(".npy"):
                continue

            basename = os.path.splitext(filename)[0]
            try:
                patient_id = int(basename.replace("patient_", ""))
            except ValueError:
                print(f"⚠️ Invalid filename format: {filename}")
                continue

            if patient_id not in label_map:
                continue

            try:
                data = np.load(os.path.join(npy_dir, filename), allow_pickle=True).item()
                data = data["data"]
            except Exception as e:
                print(f"⚠️ Error loading file {filename}: {e}")
                continue

            if not isinstance(data, np.ndarray) or data.ndim != 2:
                print(f"⚠️ Unexpected data shape in {filename}: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                continue

            # Pad or truncate to max_len
            if data.shape[0] > max_len:
                data = data[:max_len, :]
            elif data.shape[0] < max_len:
                pad_width = max_len - data.shape[0]
                data = np.pad(data, ((0, pad_width), (0, 0)), mode="constant")

            self.data.append(torch.tensor(data, dtype=torch.float32))
            self.labels.append(torch.tensor(label_map[patient_id], dtype=torch.float32))

        print(f"✅ Loaded {len(self.data)} patient samples for LOS prediction")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],   # (T, D)
            "label": self.labels[idx] # LOS in days (float)
        }
