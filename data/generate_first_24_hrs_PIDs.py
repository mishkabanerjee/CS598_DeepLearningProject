import pandas as pd
import numpy as np
import os

# Path to your imputed_stage_csv folder
imputed_folder = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\imputed_stage\imputed_stage_csv\imputed_stage\csv"

# List all CSV part files
csv_files = [os.path.join(imputed_folder, f) for f in os.listdir(imputed_folder) if f.endswith('.csv')]

# Parameters
SECONDS_IN_24_HOURS = 24 * 60 * 60
interval = 300  # 5 minutes between samples
min_timesteps = SECONDS_IN_24_HOURS // interval

# Collect valid PIDs
valid_PIDs = []

for i, file in enumerate(csv_files):
    print(f"Processing {i+1}/{len(csv_files)}: {file}")
    df = pd.read_csv(file)

    # Ensure required columns are present
    if 'patientid' not in df.columns or 'reldatetime' not in df.columns:
        continue

    df = df.sort_values(['patientid', 'reldatetime'])

    for pid, group in df.groupby('patientid'):
        group = group.drop_duplicates(subset='reldatetime')
        if group.shape[0] >= min_timesteps:
            valid_PIDs.append(int(pid))

# Remove duplicates and sort
unique_PIDs = sorted(list(set(valid_PIDs)))

# Save result
np.save("first_24_hrs_PIDs.npy", np.array(unique_PIDs))
print(f"Saved {len(unique_PIDs)} PIDs to first_24_hrs_PIDs.npy")
