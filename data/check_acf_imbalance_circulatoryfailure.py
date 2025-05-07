import os
import pandas as pd

# === Paths ===
LABELS_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\circulatory_failure\circulatory_failure_labels.csv"
ACF_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_newbatch"

# === Load Labels ===
df = pd.read_csv(LABELS_PATH)
df["patientid"] = df["patientid"].astype(int)
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
print(f"🟢 No Circulatory Failure (label=0): {negative} ({(negative/total)*100:.2f}%)")
print(f"🔴 Circulatory Failure (label=1): {positive} ({(positive/total)*100:.2f}%)")
