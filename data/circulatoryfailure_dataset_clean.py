import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HiRIDCirculatoryFailureDatasetClean(Dataset):
    def __init__(self, npy_dir, label_path, max_len=256):
        self.data = []
        self.labels = []

        label_df = pd.read_csv(label_path)
        label_map = dict(zip(label_df["patientid"], label_df["label"]))

        for filename in os.listdir(npy_dir):
            if not filename.startswith("patient_") or not filename.endswith(".npy") or filename.endswith("_M.npy"):
                continue

            try:
                patient_id = int(filename.replace("patient_", "").replace(".npy", ""))
            except ValueError:
                continue

            if patient_id not in label_map:
                continue

            filepath = os.path.join(npy_dir, filename)
            try:
                sample = np.load(filepath, allow_pickle=True)

                if isinstance(sample, np.ndarray) and sample.dtype == object:
                    sample = sample.item()
                    data = sample["data"]
                elif isinstance(sample, np.ndarray) and sample.ndim == 2:
                    data = sample
                else:
                    print(f"⚠️ Unrecognized format in {filename}")
                    continue

                if not isinstance(data, np.ndarray) or data.ndim != 2:
                    print(f"⚠️ Invalid shape in {filename}")
                    continue

                # Pad or truncate
                if data.shape[0] > max_len:
                    data = data[:max_len, :]
                elif data.shape[0] < max_len:
                    pad_width = max_len - data.shape[0]
                    data = np.pad(data, ((0, pad_width), (0, 0)), mode="constant")

                data = np.nan_to_num(data)

                self.data.append(torch.tensor(data, dtype=torch.float32))
                self.labels.append(torch.tensor(label_map[patient_id], dtype=torch.float32))

            except Exception as e:
                print(f"⚠️ Error loading file {filename}: {e}")

        print(f"✅ Loaded {len(self.data)} circulatory failure samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx]
        }
