import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.mortality_dataset import HiRIDMortalityDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# ----------------------------
# Setup
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDMortalityDataset(
    npy_dir=PretrainConfig.data_dir,
    general_table_path=r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv",
    max_len=PretrainConfig.max_len,
)

# Simple train/test split
n = len(dataset)
n_train = int(0.8 * n)
n_test = n - n_train
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load model components
encoder = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout
).encoder.to(device)

classifier = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)

# Load weights
checkpoint = torch.load("ckpt/trace_finetune_mortality.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
classifier.load_state_dict(checkpoint["classifier"])

encoder.eval()
classifier.eval()

# ----------------------------
# Evaluation
# ----------------------------

all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        encoded = encoder(data)           # (B, T, H)
        pooled = encoded.mean(dim=1)      # (B, H)
        logits = classifier(pooled)       # (B, 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

# Convert to numpy
all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()

print("Unique labels:", set(all_labels))
print("Label distribution:", np.unique(all_labels, return_counts=True))
print("Prob shape:", all_probs.shape)
print("Label shape:", all_labels.shape)

# Compute metrics
auroc = roc_auc_score(all_labels, all_probs)
accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(float))

print(f"✅ AUROC: {auroc:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Mortality Prediction")
plt.legend()
plt.grid()
plt.savefig("mortality_roc_curve.png")
plt.show()
