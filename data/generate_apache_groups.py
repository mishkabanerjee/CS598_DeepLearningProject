import os
import numpy as np
import pandas as pd
from typing import Dict

# === Config ===
OBS_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\raw_stage\observation_tables_csv"
PIDs_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\first_24_hrs_PIDs.npy"
SAVE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy_apache"

# === Constants ===
APACHE_II_ID = 9990002
APACHE_IV_ID = 9990004

# === Load PIDs ===
PIDs = set(np.load(PIDs_PATH))
group_map: Dict[int, int] = {}

# === Process each CSV file one at a time ===
for filename in os.listdir(OBS_PATH):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(OBS_PATH, filename))
        df = df[df["patientid"].isin(PIDs)]
        for pid, gdf in df.groupby("patientid"):
            if pid in group_map:
                continue
            apache_val = -1
            if not gdf[gdf["variableid"] == APACHE_IV_ID].empty:
                apache_val = int(gdf[gdf["variableid"] == APACHE_IV_ID]["value"].mode()[0])
            elif not gdf[gdf["variableid"] == APACHE_II_ID].empty:
                apache_val = int(gdf[gdf["variableid"] == APACHE_II_ID]["value"].mode()[0])
            group_map[pid] = apache_val

# === Match PIDs order and vectorize ===
PIDs_list = np.load(PIDs_PATH)
apache_groups = np.array([group_map.get(pid, -1) for pid in PIDs_list])

# === Split ===
split_idx = int(0.8 * len(PIDs_list))
train_PIDs = PIDs_list[:split_idx]
test_PIDs = PIDs_list[split_idx:]
train_groups = apache_groups[:split_idx]
test_groups = apache_groups[split_idx:]

# === Save ===
os.makedirs(SAVE_PATH, exist_ok=True)
np.save(os.path.join(SAVE_PATH, "train_first_24_hrs_PIDs.npy"), train_PIDs)
np.save(os.path.join(SAVE_PATH, "TEST_first_24_hrs_PIDs.npy"), test_PIDs)
np.save(os.path.join(SAVE_PATH, "train_Apache_Groups.npy"), train_groups)
np.save(os.path.join(SAVE_PATH, "TEST_Apache_Groups.npy"), test_groups)
print("Saved successfully to:", SAVE_PATH)
