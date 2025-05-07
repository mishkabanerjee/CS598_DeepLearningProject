import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore

# Path to your .npy files
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy_old"
NLAGS = 10  # Number of lags to compute

def plot_acf_for_patient(file_path, num_features=5):
    raw = np.load(file_path, allow_pickle=True)

    # Handle older object-dict format
    if isinstance(raw, np.ndarray) and raw.shape == ():  # 0-D object array
        try:
            data = raw.item().get("data", None)
            if data is None:
                raise ValueError("Missing 'data' key in dict.")
        except Exception as e:
            print(f"❌ Skipping {file_path}: Failed to extract 'data' from dict format. {e}")
            return
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        data = raw
    else:
        print(f"❌ Skipping {file_path}: Unknown format.")
        return

    data = np.nan_to_num(data)
    T, D = data.shape

    plt.figure(figsize=(15, 5))
    for i in range(min(num_features, D)):
        ts = zscore(data[:, i])
        try:
            acf_vals = acf(ts, nlags=10, fft=True)
            plt.plot(acf_vals, label=f'Feature {i}')
        except Exception as e:
            print(f"⚠️ Skipping feature {i} due to error: {e}")
    
    plt.title(f"ACF of first {min(num_features, D)} features in {os.path.basename(file_path)}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Pick a few random .npy files ===
for fname in os.listdir(NPY_DIR):
    if fname.endswith('.npy') and '_M' not in fname:
        print(f"🔍 Plotting ACF for {fname}")
        plot_acf_for_patient(os.path.join(NPY_DIR, fname))
        #break  # Remove break to plot more files