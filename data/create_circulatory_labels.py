import os
import pandas as pd

# --- Configure paths ---
pharma_dir = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\raw_stage\pharma_records_csv\pharma_records\csv"
output_path = r"data/circulatory_failure_labels.csv"  # Save relative to project root

# --- List of pharmaids indicating circulatory failure ---
vasoactive_pharma_ids = {
    1000462, 1000656, 1000657, 1000658,  # Noradrenalin
    71, 1000750, 1000649, 1000650, 1000655,  # Adrenalin
    426, 1000441,  # Dobutrex
    112, 113  # Vasopressin
}

# --- Read and aggregate all pharma records ---
all_patient_ids = set()
affected_patient_ids = set()

for filename in os.listdir(pharma_dir):
    if filename.endswith(".csv") and filename.startswith("part-"):
        full_path = os.path.join(pharma_dir, filename)
        df = pd.read_csv(full_path, usecols=["patientid", "pharmaid"])

        all_patient_ids.update(df["patientid"].unique())
        filtered = df[df["pharmaid"].isin(vasoactive_pharma_ids)]
        affected_patient_ids.update(filtered["patientid"].unique())

# --- Create labels ---
data = []
for pid in all_patient_ids:
    label = 1 if pid in affected_patient_ids else 0
    data.append({"patientid": pid, "label": label})

label_df = pd.DataFrame(data)
label_df.to_csv(output_path, index=False)

print(f"✅ Saved {len(label_df)} patient labels to {output_path}")
