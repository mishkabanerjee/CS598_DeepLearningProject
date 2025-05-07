import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.los_dataset_clean import HiRIDLOSDatasetClean

# === Paths ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\length_of_stay\los_labels.csv"
MODEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_los.pt"

WINDOW_SIZE = 12
ENCODING_DIM = 10
BATCH_SIZE = 64
NUM_CLASSES = 3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HiRIDLOSDatasetClean(BASE_NPY_DIR, LABEL_PATH, max_len=WINDOW_SIZE)
    test_size = int(0.2 * len(dataset))
    test_set = Subset(dataset, range(test_size))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    encoder = CausalCNNEncoder(
        in_channels=dataset[0]["data"].shape[1],
        out_channels=64,
        depth=4,
        reduced_size=4,
        encoding_size=ENCODING_DIM,
        kernel_size=3,
        window_size=WINDOW_SIZE
    ).to(device)

    classifier = torch.nn.Linear(ENCODING_DIM, NUM_CLASSES).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])

    encoder.eval()
    classifier.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"].to(device)
            labels = batch["label"].cpu()
            encoded = encoder(data)
            logits = classifier(encoded)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ LOS Accuracy: {acc:.4f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - LOS")
    plt.tight_layout()
    plt.savefig("causalcnn_los_confusion_matrix.png")
    print("📉 Confusion matrix saved as 'causalcnn_los_confusion_matrix.png'")

    # === ROC Curve (One-vs-Rest) ===
    plt.figure(figsize=(6, 5))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUROC={auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LOS (CausalCNN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("causalcnn_los_roc.png")
    print("📈 ROC curve saved as 'causalcnn_los_roc.png'")

    # === Precision-Recall Curve ===
    plt.figure(figsize=(6, 5))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve((all_labels == i).astype(int), all_probs[:, i])
        plt.plot(recall, precision, label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - LOS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("causalcnn_los_pr_curve.png")
    print("📊 PR curve saved as 'causalcnn_los_pr_curve.png'")

if __name__ == "__main__":
    main()
