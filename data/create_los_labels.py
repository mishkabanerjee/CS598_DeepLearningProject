import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\npy"
OUTPUT_PATH = "data/los_labels.csv"

# === LOS BIN EDGES (in days) for classification ===
# You can adjust these according to TRACE paper's bucket if needed
los_bins = [0, 1, 3, 7, 14, 30, 1000]  # LOS classes: 0-1d, 1-3d, 3-7d, 7-14d, 14-30d, >30d

# === LOS Extraction ===
los_records = []

print("📥 Scanning .npy files to calculate LOS...")

for filename in tqdm(os.listdir(NPY_DIR)):
    if not filename.endswith(".npy") or not filename.startswith("patient_"):
        continue

    patient_id = int(filename.replace("patient_", "").replace(".npy", ""))
    try:
        sample = np.load(os.path.join(NPY_DIR, filename), allow_pickle=True).item()
        time_array = sample.get("time")

        if time_array is None or len(time_array) == 0:
            continue

        first_time = time_array[0]
        last_time = time_array[-1]

        los_hours = (last_time - first_time) / 3600
        los_days = los_hours / 24

        # Assign class based on los_bins
        los_class = np.digitize([los_days], los_bins)[0] - 1  # class 0 to 5

        los_records.append({
            "patientid": patient_id,
            "los_days": los_days,
            "label": los_class
        })
    except Exception as e:
        print(f"⚠️ Failed to process {filename}: {e}")

# === Save to CSV ===
df = pd.DataFrame(los_records)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(df)} LOS labels to {OUTPUT_PATH}")
