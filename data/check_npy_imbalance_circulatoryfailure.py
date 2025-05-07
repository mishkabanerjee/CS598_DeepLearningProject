import os
import pandas as pd

# === Paths ===
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABELS_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\circulatory_failure\circulatory_failure_labels.csv"

# === Load labels ===
df = pd.read_csv(LABELS_PATH)
label_map = {int(row["patientid"]): int(row["label"]) for _, row in df.iterrows()}

# === Check which .npy files are available
available_files = [f for f in os.listdir(NPY_DIR) if f.endswith(".npy") and f.startswith("patient_")]
available_ids = [int(f.split("_")[1].replace(".npy", "")) for f in available_files]

# === Count label distribution
counts = {0: 0, 1: 0}
total = 0
for pid in available_ids:
    if pid in label_map:
        label = label_map[pid]
        counts[label] += 1
        total += 1

# === Print results
print(f"✅ Total patients used in causal CNN: {total}")
print(f"🟢 No Circulatory Failure (label=0): {counts[0]} ({counts[0]/total:.2%})")
print(f"🔴 Circulatory Failure (label=1): {counts[1]} ({counts[1]/total:.2%})")
