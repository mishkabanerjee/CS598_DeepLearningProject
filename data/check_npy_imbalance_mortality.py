import os
import pandas as pd
from collections import Counter

NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"

# Step 1: Get patient IDs from .npy files
npy_patients = {
    int(f.replace("patient_", "").replace(".npy", ""))
    for f in os.listdir(NPY_DIR)
    if f.startswith("patient_") and f.endswith(".npy") and not f.endswith("_M.npy")
}

# Step 2: Load general table and normalize status
df = pd.read_csv(GENERAL_TABLE_PATH)
df["discharge_status"] = df["discharge_status"].str.lower().fillna("")

# Step 3: Filter only rows corresponding to available .npy patients
df = df[df["patientid"].isin(npy_patients)]
df = df[df["discharge_status"].isin(["alive", "dead"])]
df["label_mortality"] = df["discharge_status"].map({"alive": 0, "dead": 1})

# Step 4: Compute class counts
counts = Counter(df["label_mortality"])
total = sum(counts.values())
pos_pct = counts[1] / total * 100
neg_pct = counts[0] / total * 100

# Step 5: Print results
print(f"✅ Total patients used in causal CNN: {total}")
print(f"🟢 Alive (label=0): {counts[0]} ({neg_pct:.2f}%)")
print(f"🔴 Dead  (label=1): {counts[1]} ({pos_pct:.2f}%)")
