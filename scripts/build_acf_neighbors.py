import os
import json
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore

# === CONFIG ===
DATA_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
SAVE_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"
WINDOW_SIZE = 12
POS_THRESH = 0.3
NEG_THRESH = 0.1
NLAGS = 5

# Less strict ACF similarity thresholds
MIN_CORR = 0.1
MAX_DIFF = 0.3
MIN_OVERLAP = 2

os.makedirs(SAVE_DIR, exist_ok=True)

def acf_sim(x, y):
    try:
        ax = acf(zscore(x), nlags=NLAGS, fft=True)
        ay = acf(zscore(y), nlags=NLAGS, fft=True)
        # Check how many overlapping lags have correlation above MIN_CORR and within MAX_DIFF
        overlap = sum(
            abs(ax[i] - ay[i]) <= MAX_DIFF and min(abs(ax[i]), abs(ay[i])) >= MIN_CORR
            for i in range(1, min(len(ax), len(ay)))
        )
        return overlap
    except Exception:
        return 0

def compute_acf_neighbors(data):
    T, D = data.shape
    result = {}
    for i in range(T - WINDOW_SIZE):
        anchor = data[i:i+WINDOW_SIZE]
        pos, neg = [], []
        for j in range(T - WINDOW_SIZE):
            if i == j:
                continue
            candidate = data[j:j+WINDOW_SIZE]
            overlap_scores = [acf_sim(anchor[:, d], candidate[:, d]) for d in range(D)]
            if np.mean(overlap_scores) >= MIN_OVERLAP:
                pos.append(j)
            elif 0 < np.mean(overlap_scores) < MIN_OVERLAP:
                neg.append(j)
        result[i] = {"pos": pos, "neg": neg}
    return result

def get_sorted_npy_files():
    def valid(f): return f.endswith(".npy") and not f.endswith("_M.npy") and f.startswith("patient_")
    files = [f for f in os.listdir(DATA_DIR) if valid(f)]
    return sorted(files, key=lambda f: os.path.getsize(os.path.join(DATA_DIR, f)))

def main():
    files = get_sorted_npy_files()
    for file in tqdm(files, desc="Processing patients"):
        fpath = os.path.join(DATA_DIR, file)
        json_path = os.path.join(SAVE_DIR, file.replace(".npy", ".json"))

        if os.path.exists(json_path):
            continue

        try:
            raw = np.load(fpath, allow_pickle=True)

            # Handle dict-based npy
            if isinstance(raw, np.ndarray) and raw.dtype == object:
                try:
                    raw = raw.item()
                except Exception:
                    print(f"⚠️ Skipping {file}: unable to unpack .item()")
                    continue
                if not isinstance(raw, dict) or "data" not in raw:
                    print(f"⚠️ Skipping {file}: no 'data' key found.")
                    continue
                data = raw["data"]

            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                data = raw

            else:
                print(f"⚠️ Skipping {file}: Unrecognized .npy structure.")
                continue

            if not isinstance(data, np.ndarray) or data.ndim != 2:
                print(f"⚠️ Skipping {file}: 'data' is not a valid 2D array.")
                continue

            acf_map = compute_acf_neighbors(data)
            with open(json_path, "w") as f:
                json.dump(acf_map, f)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    main()
