import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HiRIDLOSDatasetClean(Dataset):
    def __init__(self, npy_dir, label_path, max_len=256):
        self.data = []
        self.labels = []

        label_df = pd.read_csv(label_path)
        label_map = dict(zip(label_df["patientid"], label_df["los_days"]))

        for filename in os.listdir(npy_dir):
            if not filename.endswith(".npy") or "_M" in filename:
                continue

            basename = os.path.splitext(filename)[0]
            try:
                patient_id = int(basename.replace("patient_", ""))
            except ValueError:
                print(f"⚠️ Invalid filename format: {filename}")
                continue

            if patient_id not in label_map:
                continue

            filepath = os.path.join(npy_dir, filename)
            try:
                sample = np.load(filepath, allow_pickle=True)
                if isinstance(sample, dict) and "data" in sample:
                    data = sample["data"]
                elif isinstance(sample, np.ndarray) and sample.size == 1 and isinstance(sample.item(), dict):
                    sample_dict = sample.item()
                    if "data" in sample_dict:
                        data = sample_dict["data"]
                    else:
                        print(f"⚠️ Missing 'data' key in wrapped dict in {filename}, skipping.")
                        continue
                elif isinstance(sample, np.ndarray) and sample.ndim == 2:
                    data = sample
                else:
                    print(f"⚠️ Unexpected format in {filename}, skipping.")
                    continue
            except Exception as e:
                print(f"⚠️ Error loading file {filename}: {e}")
                continue

            if not isinstance(data, np.ndarray) or data.ndim != 2:
                print(f"⚠️ Unexpected data shape in {filename}: {getattr(data, 'shape', 'N/A')}")
                continue

            # Pad or truncate
            if data.shape[0] > max_len:
                data = data[:max_len]
            elif data.shape[0] < max_len:
                pad_len = max_len - data.shape[0]
                data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")

            data = np.nan_to_num(data)

            # Bin LOS into 3 categories
            los_days = label_map[patient_id]
            if los_days <= 3:
                label = 0
            elif los_days <= 7:
                label = 1
            else:
                label = 2

            self.data.append(torch.tensor(data, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))

        print(f"✅ Loaded {len(self.data)} samples for LOS classification")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],   # (T, D)
            "label": self.labels[idx] # class 0, 1, or 2
        }
