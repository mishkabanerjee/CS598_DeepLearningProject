import os

# === File Paths ===

# Path to your patient data (.npy files)
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"

# Mask file naming convention (_M.npy means mask of patient_X.npy)
MASK_SUFFIX = "_M.npy"

# Where encoded embeddings will be saved
ENCODING_OUTPUT_DIR = os.path.join("data", "encoded")

# ACF Neighbor Map Directory (optional, only used for pretraining)
ACF_NEIGHBOR_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"

# === Label Files for Downstream Tasks ===
LABEL_PATHS = {
    "mortality": {
        "train_ids": os.path.join("data", "mortality", "train_patient_ids_mortality.npy"),
        "train_labels": os.path.join("data", "mortality", "train_labels_mortality.npy"),
        "test_ids": os.path.join("data", "mortality", "test_patient_ids_mortality.npy"),
        "test_labels": os.path.join("data", "mortality", "test_labels_mortality.npy"),
    },
    "circulatory_failure": {
        "train_ids": os.path.join("data", "circulatory_failure", "train_patient_ids_circ_failure.npy"),
        "train_labels": os.path.join("data", "circulatory_failure", "train_labels_circ_failure.npy"),
        "test_ids": os.path.join("data", "circulatory_failure", "test_patient_ids_circ_failure.npy"),
        "test_labels": os.path.join("data", "circulatory_failure", "test_labels_circ_failure.npy"),
    },
    "length_of_stay": {
    "train_ids": os.path.join("data", "length_of_stay", "train_patient_ids_los.npy"),
    "train_labels": os.path.join("data", "length_of_stay", "train_labels_los.npy"),
    "test_ids": os.path.join("data", "length_of_stay", "test_patient_ids_los.npy"),
    "test_labels": os.path.join("data", "length_of_stay", "test_labels_los.npy"),
    },
}

# === General Hyperparameters ===
WINDOW_SIZE = 12
ENCODING_DIM = 10  # for random encoder or pretrained encoder output size
BATCH_SIZE = 128
SEED = 42
