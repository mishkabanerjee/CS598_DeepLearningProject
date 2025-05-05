import os
import numpy as np

np.random.seed(42)
npy_dir = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
#files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
files = [f for f in os.listdir(npy_dir) if f.endswith('.npy') and "patient_" in f and f.count("_") == 1]


patient_ids = [int(f.replace("patient_", "").replace(".npy", "")) for f in files]
patient_ids = sorted(patient_ids)
np.random.shuffle(patient_ids)

split = int(0.8 * len(patient_ids))
train_ids = patient_ids[:split]
test_ids = patient_ids[split:]

np.save("train_patient_ids.npy", train_ids)
np.save("test_patient_ids.npy", test_ids)

print(f"✅ Saved {len(train_ids)} train and {len(test_ids)} test IDs.")
