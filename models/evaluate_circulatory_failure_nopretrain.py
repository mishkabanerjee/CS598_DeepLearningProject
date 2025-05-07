import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from models.trace_model import TRACEModel
from data.circulatoryfailure_dataset import HiRIDCirculatoryFailureDataset
from models.train_pretrain_config import PretrainConfig
from models.classifier_head import ClassificationHead

# ----------------------------
# Setup
# ----------------------------

print("📥 Loading circulatory failure dataset...")
dataset = HiRIDCirculatoryFailureDataset(
    npy_dir=PretrainConfig.data_dir,
    label_path="data/circulatory_failure_labels.csv",
    max_len=PretrainConfig.max_len,
)

print(f"✅ Loaded {len(dataset)} patient samples for circulatory failure classification")

# Simple train/test split
n = len(dataset)
n_train = int(0.8 * n)
n_test = n - n_train
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model components
encoder = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout
).encoder.to(device)

classifier = ClassificationHead(
    input_dim=PretrainConfig.hidden_dim,
    hidden_dim=64,  # Match the same hidden_dim used in training
    output_dim=1,
    dropout=PretrainConfig.dropout,
).to(device)


# Load checkpoint
ckpt = torch.load("ckpt/trace_finetune_circulatory_failure_nopretrain.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
classifier.load_state_dict(ckpt["classifier"])

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
plt.title("ROC Curve for Circulatory Failure (No Pretrain)")
plt.legend()
plt.grid()
plt.savefig("circulatory_roc_nopretrain.png")
plt.show()
