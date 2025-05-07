import os
import pandas as pd

# === Paths ===
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
ACF_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"

# === Load Mortality Labels ===
df = pd.read_csv(GENERAL_TABLE_PATH)
df["patientid"] = df["patientid"].astype(int)

# Assign label: 1 = dead, 0 = alive
df["label"] = df["discharge_status"].str.lower().apply(lambda x: 1 if x == "dead" else 0)
label_map = dict(zip(df["patientid"], df["label"]))

# === Only ACF Patients ===
json_patients = [
    int(fname.replace("patient_", "").replace(".json", ""))
    for fname in os.listdir(ACF_DIR) if fname.endswith(".json")
]

# === Count Labels in ACF Patients Only ===
positive = sum(label_map.get(pid, 0) == 1 for pid in json_patients)
negative = sum(label_map.get(pid, 0) == 0 for pid in json_patients)
total = positive + negative

print(f"✅ Total ACF patients: {total}")
print(f"🟢 Alive (label=0): {negative} ({(negative/total)*100:.2f}%)")
print(f"🔴 Dead  (label=1): {positive} ({(positive/total)*100:.2f}%)")
