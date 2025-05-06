import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
MAX_PATIENTS = 1000  # For speed

all_data = []

print("📥 Loading patient matrices ...")
for i, fname in enumerate(os.listdir(NPY_DIR)):
    if fname.endswith(".npy"):
        arr = np.load(os.path.join(NPY_DIR, fname), allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            all_data.append(arr)
        if len(all_data) >= MAX_PATIENTS:
            break

# Stack along time (T) dimension
combined = np.concatenate(all_data, axis=0)  # shape: [Total_T, D]

print(f"✅ Combined shape: {combined.shape}")
D = combined.shape[1]

# Compute missingness
nan_counts = np.isnan(combined).sum(axis=0)
total_counts = combined.shape[0]
missing_ratios = nan_counts / total_counts

# Plot
plt.figure(figsize=(10, 5))
plt.bar(range(D), missing_ratios, color='salmon')
plt.title("📉 Variable-Wise Missingness Ratio")
plt.xlabel("Feature Index")
plt.ylabel("Missing Ratio")
plt.xticks(range(D))
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Optional table for report
df = pd.DataFrame({
    "Feature Index": list(range(D)),
    "Missing Ratio": missing_ratios
})
print(df.to_markdown(index=False))
