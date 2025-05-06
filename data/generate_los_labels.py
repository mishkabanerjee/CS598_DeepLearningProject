import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# === Paths ===
CSV_PATH = "data/length_of_stay/los_labels.csv"
OUT_DIR = "data/length_of_stay"
os.makedirs(OUT_DIR, exist_ok=True)

# === Load LOS labels ===
df = pd.read_csv(CSV_PATH)

# Ensure it's sorted consistently
df = df.sort_values("patientid")

# === Split into Train/Test ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# === Save as .npy ===
np.save(os.path.join(OUT_DIR, "train_patient_ids_los.npy"), train_df["patientid"].values)
np.save(os.path.join(OUT_DIR, "train_labels_los.npy"), train_df["label"].values)
np.save(os.path.join(OUT_DIR, "test_patient_ids_los.npy"), test_df["patientid"].values)
np.save(os.path.join(OUT_DIR, "test_labels_los.npy"), test_df["label"].values)

print(f"✅ Saved {len(train_df)} train and {len(test_df)} test LOS labels to: {OUT_DIR}")
