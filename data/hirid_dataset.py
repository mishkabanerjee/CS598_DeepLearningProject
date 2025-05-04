import os
import numpy as np
import torch
from torch.utils.data import Dataset

class HiRIDDataset(Dataset):
    def __init__(self, data_dir, max_len=256):
        """
        Args:
            data_dir (str): Path where patient .npy files are stored
            max_len (int): Maximum time steps per patient (truncate or pad)
        """
        self.data_dir = data_dir
        self.files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')
        ])
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        sample = np.load(sample_path, allow_pickle=True).item()
        
        data = sample['data']   # shape (T, num_features)
        time = sample['time']   # shape (T,)

        # --- Preprocessing ---
        # Truncate or pad to max_len
        T, D = data.shape
        if T >= self.max_len:
            data = data[:self.max_len]
            time = time[:self.max_len]
        else:
            pad_length = self.max_len - T
            data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=np.nan)
            time = np.pad(time, (0, pad_length), mode='constant', constant_values=0)

        # Replace nans with zeros (alternative: keep mask separately if needed)
        data = np.nan_to_num(data)

        # --- Convert to torch tensors ---
        data = torch.tensor(data, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)

        return {
            "data": data,         # (max_len, num_features)
            "time": time,         # (max_len,)
            "mask": (data != 0).float()  # optional: mask where original data was nonzero
        }
