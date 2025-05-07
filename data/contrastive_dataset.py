import os
import json
import numpy as np
from torch.utils.data import Dataset

class ContrastiveACFDataset(Dataset):
    def __init__(self, npy_dir, acf_dir, window_size=12):
        self.npy_dir = npy_dir
        self.acf_dir = acf_dir
        self.window_size = window_size
        self.pairs = []

        for fname in os.listdir(acf_dir):
            if not fname.endswith(".json"):
                continue

            patient_id = fname.replace(".json", "")
            npy_path = os.path.join(npy_dir, f"{patient_id}.npy")
            if not os.path.exists(npy_path):
                continue

            # Load and validate npy file
            data = np.load(npy_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item().get("data", None)
            if data is None or data.ndim != 2:
                continue

            T = data.shape[0]
            with open(os.path.join(acf_dir, fname), "r") as f:
                acf_map = json.load(f)

            for anchor_idx, entry in acf_map.items():
                anchor_idx = int(anchor_idx)
                for pos_idx in entry.get("pos", []):
                    pos_idx = int(pos_idx)
                    if pos_idx + window_size <= T and anchor_idx + window_size <= T:
                        self.pairs.append((patient_id, anchor_idx, pos_idx, 1))
                for neg_idx in entry.get("neg", []):
                    neg_idx = int(neg_idx)
                    if neg_idx + window_size <= T and anchor_idx + window_size <= T:
                        self.pairs.append((patient_id, anchor_idx, neg_idx, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        patient_id, anchor_idx, other_idx, is_positive = self.pairs[idx]
        npy_path = os.path.join(self.npy_dir, f"{patient_id}.npy")
        data = np.load(npy_path, allow_pickle=True)

        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item().get("data", None)

        anchor = data[anchor_idx: anchor_idx + self.window_size]
        other = data[other_idx: other_idx + self.window_size]

        return {
            "anchor": anchor.astype(np.float32),
            "other": other.astype(np.float32),
            "label": np.array(is_positive, dtype=np.float32),
        }
