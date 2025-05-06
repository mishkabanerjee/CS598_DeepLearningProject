import os
import numpy as np
import matplotlib.pyplot as plt

# === Path to your .npy files ===
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"

lengths = []

# === Loop through files ===
for fname in os.listdir(NPY_DIR):
    if fname.endswith(".npy"):
        fpath = os.path.join(NPY_DIR, fname)
        try:
            arr = np.load(fpath, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                lengths.append(arr.shape[0])
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")

# === Plot histogram ===
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title("⏱ Distribution of Patient Time Series Lengths")
plt.xlabel("Number of Time Steps (T)")
plt.ylabel("Number of Patients")
plt.axvline(256, color='red', linestyle='--', label='max_len = 256')
plt.legend()
plt.tight_layout()
plt.show()

# === Optional stats ===
print(f"✅ Processed {len(lengths)} patients")
print(f"📊 Median length: {np.median(lengths):.0f}")
print(f"📈 Max length   : {np.max(lengths)}")
print(f"📉 Min length   : {np.min(lengths)}")
print(f"🧮 % <= 256     : {(np.sum(np.array(lengths) <= 256) / len(lengths)) * 100:.2f}%")
