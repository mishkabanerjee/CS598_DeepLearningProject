import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
import seaborn as sns

from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.circulatoryfailure_dataset import HiRIDCirculatoryFailureDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# ----------------------------
# Setup
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDCirculatoryFailureDataset(
    npy_dir=PretrainConfig.data_dir,
    label_path="data/circulatory_failure/circulatory_failure_labels.csv",
    max_len=PretrainConfig.max_len,
)

print(f"✅ Loaded {len(dataset)} patient samples for circulatory failure classification")

# Split into test set
n = len(dataset)
n_train = int(0.8 * n)
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42))

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
ckpt = torch.load("ckpt/trace_finetune_circulatory_failure.pt", map_location=device)
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

        encoded = encoder(data)
        pooled = encoded.mean(dim=1)
        logits = classifier(pooled)
        probs = torch.sigmoid(logits).squeeze(-1)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

all_probs = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()
all_preds = (all_probs > 0.5).astype(float)

# Metrics
auroc = roc_auc_score(all_labels, all_probs)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print metrics
print(f"✅ AUROC: {auroc:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC curve plot
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Circulatory Failure Prediction")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("circulatory_failure_roc_curve.png")
plt.show()
