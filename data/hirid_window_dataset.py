import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import zscore
from statsmodels.tsa.stattools import acf

class HiRIDWindowDataset(Dataset):
    def __init__(self, data_dir, window_size=12, acf_pos_thresh=0.6, acf_neg_thresh=0.1, num_negatives=5):
        self.data_dir = data_dir
        self.window_size = window_size
        self.acf_pos_thresh = acf_pos_thresh
        self.acf_neg_thresh = acf_neg_thresh
        self.num_negatives = num_negatives

        self.patient_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.window_tuples = []  # List of (file_path, window_start_idx)

        for fpath in self.patient_files:
            data = np.load(fpath)  # shape: (T, D)
            T = data.shape[0]
            for t in range(T - window_size):
                self.window_tuples.append((fpath, t))

    def __len__(self):
        return len(self.window_tuples)

    def _get_window(self, data, start):
        return data[start : start + self.window_size]

    def _find_neighbors(self, data, idx):
        anchor = self._get_window(data, idx)
        T = data.shape[0]

        pos_indices = []
        neg_indices = []

        for t in range(T - self.window_size):
            if t == idx:
                continue
            candidate = self._get_window(data, t)

            # compute ACF similarity (mean of ACF correlations across features)
            sim = np.mean([
                self._acf_sim(anchor[:, d], candidate[:, d])
                for d in range(data.shape[1])
            ])

            if sim >= self.acf_pos_thresh:
                pos_indices.append(t)
            elif sim <= self.acf_neg_thresh:
                neg_indices.append(t)

        return pos_indices, neg_indices

    def _acf_sim(self, x, y, nlags=5):
        try:
            ax = acf(zscore(x), nlags=nlags, fft=True)
            ay = acf(zscore(y), nlags=nlags, fft=True)
            return np.corrcoef(ax, ay)[0, 1]
        except:
            return 0.0

    def __getitem__(self, index):
        fpath, idx = self.window_tuples[index]
        data = np.load(fpath)

        anchor = self._get_window(data, idx)
        pos_indices, neg_indices = self._find_neighbors(data, idx)

        if not pos_indices:
            pos = anchor.copy()  # fallback: use self
        else:
            pos = self._get_window(data, np.random.choice(pos_indices))

        if len(neg_indices) < self.num_negatives:
            negs = [self._get_window(data, np.random.choice(range(data.shape[0] - self.window_size)))
                    for _ in range(self.num_negatives)]
        else:
            negs = [self._get_window(data, i) for i in np.random.choice(neg_indices, self.num_negatives, replace=False)]

        return (
            torch.tensor(anchor, dtype=torch.float),        # z_anchor
            torch.tensor(pos, dtype=torch.float),           # z_pos
            torch.stack([torch.tensor(n, dtype=torch.float) for n in negs])  # z_negatives: (K, T, D)
        )
