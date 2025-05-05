import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
HIRID_BASE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid"
CSV_FOLDER = os.path.join(HIRID_BASE_PATH, "imputed_stage", "imputed_stage_csv", "imputed_stage", "csv")
VAR_REF_PATH = os.path.join(HIRID_BASE_PATH, "reference_data", "hirid_variable_reference_preprocessed.csv")
NPY_FOLDER = os.path.join(HIRID_BASE_PATH, "npy")
os.makedirs(NPY_FOLDER, exist_ok=True)

# ==========================
# LOAD VARIABLE IDS
# ==========================
print("📥 Loading variable reference...")
var_ref = pd.read_csv(VAR_REF_PATH)
# Replace 'short_label' with something that definitely exists
print("📜 Available columns in var_ref:", var_ref.columns.tolist())

# Use all Variable ids that aren't empty or NaN
if "Variable id" in var_ref.columns:
    var_ids = var_ref["Variable id"].dropna().unique().tolist()
elif "id" in var_ref.columns:
    var_ids = var_ref["id"].dropna().unique().tolist()
else:
    raise ValueError("❌ Could not find a column with variable IDs in the reference file.")


# ==========================
# LOAD RAW CSVs (ORIGINAL)
# ==========================
print("📊 Reading imputed CSVs...")
all_files = [os.path.join(CSV_FOLDER, f) for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]
df_all = pd.concat([pd.read_csv(f) for f in tqdm(all_files)], ignore_index=True)
df_all = df_all[df_all["patientid"].notnull()]
df_all["patientid"] = df_all["patientid"].astype(int)
df_all = df_all.sort_values(["patientid", "reldatetime"])

# Only keep needed vars
cols_to_keep = ["patientid", "reldatetime"] + [v for v in var_ids if v in df_all.columns]
df_all = df_all[cols_to_keep]

# ==========================
# BUILD MASKS PER PATIENT
# ==========================
print("⚙️ Generating missing masks...")
grouped = df_all.groupby("patientid")

for pid, group in tqdm(grouped):
    mask_path = os.path.join(NPY_FOLDER, f"patient_{pid}_M.npy")
    if os.path.exists(mask_path):
        continue  # skip if already exists

    x_path = os.path.join(NPY_FOLDER, f"patient_{pid}.npy")
    if not os.path.exists(x_path):
        continue  # skip if no X.npy to match

    # Sort + initialize
    group = group.sort_values("reldatetime")
    T = len(group)
    variables = [v for v in var_ids if v in group.columns]
    D = len(variables)

    M = np.zeros((T, D), dtype=np.uint8)
    for i, (_, row) in enumerate(group.iterrows()):
        for j, v in enumerate(variables):
            val = row[v]
            if pd.notnull(val):
                M[i, j] = 1

    # Save mask
    np.save(mask_path, M)

print("✅ Masks generated and saved where needed.")
