import pandas as pd
import numpy as np
import os

# === Paths ===
REF_CSV = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
NPY_FOLDER = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
TRAIN_ID_FILE = "train_patient_ids.npy"
TEST_ID_FILE = "test_patient_ids.npy"

# === Load discharge status ===
print("📥 Loading general_table.csv ...")
df = pd.read_csv(REF_CSV)

# Convert to lowercase just in case
df["discharge_status"] = df["discharge_status"].str.lower()

# Map 'dead' → 1, 'alive' → 0
status_map = {"dead": 1, "alive": 0}
df["mortality"] = df["discharge_status"].map(status_map)

# Drop rows without a valid label
df = df[df["mortality"].isin([0, 1])]
mortality_map = dict(zip(df["patientid"], df["mortality"]))

# === Load patient ID splits ===
train_ids = np.load(TRAIN_ID_FILE)
test_ids = np.load(TEST_ID_FILE)

# === Create label arrays ===
train_labels = [mortality_map.get(pid, -1) for pid in train_ids]
test_labels = [mortality_map.get(pid, -1) for pid in test_ids]

# Filter out missing labels (-1)
train_ids_clean = [pid for pid, label in zip(train_ids, train_labels) if label in (0, 1)]
test_ids_clean = [pid for pid, label in zip(test_ids, test_labels) if label in (0, 1)]
train_labels_clean = [label for label in train_labels if label in (0, 1)]
test_labels_clean = [label for label in test_labels if label in (0, 1)]

# === Save outputs ===
np.save("train_patient_ids_mortality.npy", train_ids_clean)
np.save("test_patient_ids_mortality.npy", test_ids_clean)
np.save("train_labels_mortality.npy", train_labels_clean)
np.save("test_labels_mortality.npy", test_labels_clean)

print(f"✅ Saved {len(train_ids_clean)} train and {len(test_ids_clean)} test mortality labels.")
