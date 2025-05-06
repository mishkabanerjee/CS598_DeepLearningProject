import pandas as pd
import numpy as np

# === Paths ===
LABEL_CSV = "data/circulatory_failure_labels.csv"
TRAIN_IDS_PATH = "train_patient_ids.npy"
TEST_IDS_PATH = "test_patient_ids.npy"

# === Load labels ===
df = pd.read_csv(LABEL_CSV)
label_dict = dict(zip(df["patientid"], df["label"]))

# === Load patient IDs ===
train_ids = np.load(TRAIN_IDS_PATH)
test_ids = np.load(TEST_IDS_PATH)

# === Map labels ===
train_labels = [label_dict.get(pid, -1) for pid in train_ids]
test_labels = [label_dict.get(pid, -1) for pid in test_ids]

# Filter missing
train_ids_clean = [pid for pid, label in zip(train_ids, train_labels) if label in (0, 1)]
test_ids_clean = [pid for pid, label in zip(test_ids, test_labels) if label in (0, 1)]
train_labels_clean = [label for label in train_labels if label in (0, 1)]
test_labels_clean = [label for label in test_labels if label in (0, 1)]

# === Save outputs ===
np.save("train_patient_ids_circ_failure.npy", train_ids_clean)
np.save("test_patient_ids_circ_failure.npy", test_ids_clean)
np.save("train_labels_circ_failure.npy", train_labels_clean)
np.save("test_labels_circ_failure.npy", test_labels_clean)

print(f"✅ Saved {len(train_labels_clean)} train and {len(test_labels_clean)} test circulatory failure labels.")
print(f"🔍 Train positive rate: {np.mean(train_labels_clean):.2%}")
print(f"🔍 Test positive rate : {np.mean(test_labels_clean):.2%}")
