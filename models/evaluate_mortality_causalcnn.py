import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.mortality_dataset import HiRIDMortalityDataset

# === CONFIG ===
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
CHECKPOINT_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch50.pt"
WINDOW_SIZE = 12
ENCODING_DIM = 64
BATCH_SIZE = 64

# === SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDMortalityDataset(
    npy_dir=NPY_DIR,
    general_table_path=GENERAL_TABLE_PATH,
    max_len=WINDOW_SIZE  # This is correct here — we treat the whole sequence as a fixed window
)

# Split into train/test (for evaluation only)
n = len(dataset)
n_train = int(0.8 * n)
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
encoder = CausalCNNEncoder(
    in_channels=dataset[0]["data"].shape[1],
    out_channels=64,
    depth=4,
    reduced_size=16,
    encoding_size=ENCODING_DIM,
    kernel_size=3,
    window_size=WINDOW_SIZE
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
encoder.load_state_dict(checkpoint)
encoder.eval()

# Simple classifier
classifier = torch.nn.Linear(ENCODING_DIM, 1).to(device)
classifier.eval()  # No training here — assume it’s already trained if you have saved weights

# If you have saved classifier weights separately, load them here
# classifier.load_state_dict(torch.load(...))

# === EVALUATION ===
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        data = batch["data"].to(device)  # (B, T, D)
        labels = batch["label"].to(device)

        encoded = encoder(data)         # (B, H)
        logits = classifier(encoded).squeeze(-1)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

# Convert
all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

# Metrics
auroc = roc_auc_score(all_labels, all_probs)
accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(float))

print(f"\n✅ AUROC: {auroc:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Plot ROC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - CausalCNN Mortality Prediction")
plt.legend()
plt.grid()
plt.savefig("mortality_roc_causalcnn.png")
plt.show()
