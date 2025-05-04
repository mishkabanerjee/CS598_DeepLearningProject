import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================
HIRID_BASE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid"
CSV_FOLDER = os.path.join(HIRID_BASE_PATH, "imputed_stage", "imputed_stage_csv", "imputed_stage", "csv")
VAR_REF_PATH = os.path.join(HIRID_BASE_PATH, "reference_data", "hirid_variable_reference_preprocessed.csv")
OUTPUT_PATH = os.path.join(HIRID_BASE_PATH, "npy")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ================================
# LOAD VARIABLE REFERENCE
# ================================
def load_trace_variable_ids(var_ref_path):
    print("📥 Loading variable reference...")
    df = pd.read_csv(var_ref_path)
    var_to_ids = {}
    for _, row in df.iterrows():
        trace_var = row["Variable id"]
        raw_ids = str(row["raw variable ids"]).split(",")
        raw_ids = [rid.strip() for rid in raw_ids]
        var_to_ids[trace_var] = raw_ids
    return var_to_ids

# ================================
# EXTRACT MATRICES
# ================================
def extract_trace_matrices(df, var_to_ids):
    print("🧹 Filtering and reshaping to patient matrices...")
    patient_matrices = {}
    grouped = df.groupby("patientid")
    
    for pid, group in tqdm(grouped, desc="Processing patients"):
        group = group.sort_values("reldatetime")
        time_steps = group["reldatetime"].astype(int).values
        variables = [col for col in group.columns if col not in ["patientid", "reldatetime"]]
        
        matrix = np.full((len(time_steps), len(variables)), np.nan)
        for i, row in enumerate(group.itertuples(index=False)):
            for j, var in enumerate(variables):
                matrix[i, j] = getattr(row, var)

        patient_matrices[pid] = {
            "data": matrix,
            "time": time_steps,
            "variables": variables
        }

    return patient_matrices

# ================================
# MAIN
# ================================
def main():
    var_to_ids = load_trace_variable_ids(VAR_REF_PATH)

    print("📊 Loading raw HiRID CSVs...")
    all_csv_files = glob.glob(os.path.join(CSV_FOLDER, "part-*"))
    dfs = [pd.read_csv(f) for f in all_csv_files]
    full_df = pd.concat(dfs, ignore_index=True)

    # Clean column names
    full_df.columns = full_df.columns.str.strip()
    
    # Only keep variables in our reference
    trace_vars = list(var_to_ids.keys())
    cols_to_keep = ["patientid", "reldatetime"] + [var for var in trace_vars if var in full_df.columns]
    full_df = full_df[cols_to_keep]

    # Convert to patient matrices
    patient_matrices = extract_trace_matrices(full_df, var_to_ids)

    # Save each patient’s data
    print(f"💾 Saving .npy files to: {OUTPUT_PATH}")
    for pid, pdata in tqdm(patient_matrices.items(), desc="Saving"):
        save_path = os.path.join(OUTPUT_PATH, f"patient_{pid}.npy")
        np.save(save_path, pdata)

    print("✅ Conversion complete.")

# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    main()
