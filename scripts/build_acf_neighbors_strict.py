import os, json
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore

# === Config ===
DATA_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
SAVE_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"

WINDOW_SIZE = 12
POS_THRESH = 0.6
NEG_THRESH = 0.1
NLAGS = 5

def acf_sim(x, y):
    try:
        ax = acf(zscore(x), nlags=NLAGS, fft=True)
        ay = acf(zscore(y), nlags=NLAGS, fft=True)
        return np.corrcoef(ax, ay)[0, 1]
    except:
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

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Step 1: Gather all files and their lengths
    file_lengths = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".npy"):
            continue
        try:
            data = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                file_lengths.append((fname, data.shape[0]))  # (filename, T)
        except:
            continue

    # Step 2: Sort by T (number of timesteps)
    file_lengths.sort(key=lambda x: x[1])  # smallest T first
    files = [fname for fname, _ in file_lengths]

    # Optional: Limit to 100 smallest
    files = files[:1000]

    # Step 3: Process files
    for file in tqdm(files, desc="Processing patients"):
        fpath = os.path.join(DATA_DIR, file)
        json_path = os.path.join(SAVE_DIR, file.replace('.npy', '.json'))

        if os.path.exists(json_path):
            print(f"⏩ Skipping {file}: JSON already exists.")
            continue

        try:
            data = np.load(fpath, allow_pickle=True)
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                print(f"⚠️ Skipping {file}: not a 2D array.")
                continue

            print(f"✅ Processing {file} ...")
            acf_map = compute_acf_neighbors(data)

            with open(json_path, 'w') as f:
                json.dump(acf_map, f)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")


if __name__ == "__main__":
    main()
