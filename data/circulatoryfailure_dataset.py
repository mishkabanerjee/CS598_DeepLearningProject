import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HiRIDCirculatoryFailureDataset(Dataset):
    def __init__(self, npy_dir, label_path, max_len=256):
        self.data = []
        self.labels = []

        label_df = pd.read_csv(label_path)
        label_map = dict(zip(label_df["patientid"], label_df["label"]))

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
                sample = np.load(os.path.join(npy_dir, filename), allow_pickle=True).item()
                data = sample["data"]
            except Exception as e:
                print(f"⚠️ Error loading file {filename}: {e}")
                continue

            if not isinstance(data, np.ndarray) or data.ndim != 2:
                print(f"⚠️ Unexpected data shape in {filename}: {data.shape if hasattr(data, 'shape') else 'N/A'}")
                continue

            # Pad or truncate
            if data.shape[0] > max_len:
                data = data[:max_len, :]
            elif data.shape[0] < max_len:
                pad_width = max_len - data.shape[0]
                data = np.pad(data, ((0, pad_width), (0, 0)), mode="constant")

            data = np.nan_to_num(data)  # Replace NaNs

            self.data.append(torch.tensor(data, dtype=torch.float32))
            self.labels.append(torch.tensor(label_map[patient_id], dtype=torch.float32))

        print(f"✅ Loaded {len(self.data)} patient samples for circulatory failure classification")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx]
        }
