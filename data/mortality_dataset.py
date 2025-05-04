import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HiRIDMortalityDataset(Dataset):
    def __init__(self, npy_dir, general_table_path, max_len=256):
        """
        Args:
            npy_dir (str): Path to patient .npy files
            general_table_path (str): Path to general_table.csv
            max_len (int): Maximum time steps per patient
        """
        self.npy_dir = npy_dir
        self.max_len = max_len

        # Load patient mortality labels
        general_df = pd.read_csv(general_table_path)
        general_df = general_df[['patientid', 'discharge_status']].dropna()

        # Map discharge status to 0/1
        self.label_map = {
            'alive': 0,
            'dead': 1,
            'unknown': 0   # optional: treat 'unknown' as 0
        }
        general_df['label'] = general_df['discharge_status'].map(self.label_map)

        # Create patient id to label mapping
        self.patient_labels = dict(zip(general_df['patientid'], general_df['label']))

        # Find .npy files
        self.files = sorted([
            os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        sample = np.load(sample_path, allow_pickle=True).item()
        
        patient_id = int(os.path.basename(sample_path).split("_")[1].split(".")[0])

        data = sample['data']
        time = sample['time']

        # Truncate or pad
        T, D = data.shape
        if T >= self.max_len:
            data = data[:self.max_len]
            time = time[:self.max_len]
        else:
            pad_length = self.max_len - T
            data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=np.nan)
            time = np.pad(time, (0, pad_length), mode='constant', constant_values=0)

        data = np.nan_to_num(data)

        # Load label
        label = self.patient_labels.get(patient_id, 0)  # default to 0 if not found

        return {
            "data": torch.tensor(data, dtype=torch.float32),        # (max_len, num_features)
            "time": torch.tensor(time, dtype=torch.float32),         # (max_len,)
            "mask": (torch.tensor(data) != 0).float(),               # (max_len, num_features)
            "label": torch.tensor(label, dtype=torch.float32)        # scalar
        }
