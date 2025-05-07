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
POS_THRESH = 0.3   # relaxed positive similarity threshold
NEG_THRESH = 0.1  # relaxed negative similarity threshold
NLAGS = 5

MIN_CORR = 0.1
MAX_DIFF = 0.2  # or try 0.3 if still too few pairs
MIN_OVERLAP = 2


os.makedirs(SAVE_DIR, exist_ok=True)

def append_to_log(filename, log_path="processed_files.txt"):
    with open(log_path, "a") as f:
        f.write(f"{filename}\n")

def acf_sim(x, y):
    try:
        ax = acf(zscore(x), nlags=NLAGS, fft=True)
        ay = acf(zscore(y), nlags=NLAGS, fft=True)
        return np.corrcoef(ax, ay)[0, 1]
    except Exception:
        return 0.0

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
            sims = [acf_sim(anchor[:, d], candidate[:, d]) for d in range(D)]
            mean_sim = np.mean(sims)
            if mean_sim >= POS_THRESH:
                pos.append(j)
            elif mean_sim <= NEG_THRESH:
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

            # Handle array-based npy
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

            # Optional: add this only if you want logging
            # append_to_log(file)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")



if __name__ == "__main__":
    main()
