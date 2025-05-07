import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, auc
)

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.circulatoryfailure_dataset_clean import HiRIDCirculatoryFailureDatasetClean

# === Paths ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\circulatory_failure\circulatory_failure_labels.csv"
MODEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_circulatory.pt"

# === Hyperparameters ===
WINDOW_SIZE = 12
ENCODING_DIM = 10
BATCH_SIZE = 64

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HiRIDCirculatoryFailureDatasetClean(BASE_NPY_DIR, LABEL_PATH, max_len=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    encoder = CausalCNNEncoder(
        in_channels=dataset[0]["data"].shape[1],
        out_channels=64,
        depth=4,
        reduced_size=4,
        encoding_size=ENCODING_DIM,
        kernel_size=3,
        window_size=WINDOW_SIZE
    ).to(device)

    classifier = torch.nn.Linear(ENCODING_DIM, 1).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    encoder.eval()
    classifier.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            labels = batch["label"].cpu()
            encoded = encoder(data)
            probs = torch.sigmoid(classifier(encoded)).view(-1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auroc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(all_labels, (all_probs > 0.5).astype(int))

    print(f"\n✅ Evaluation Results on Circulatory Failure Task")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")

    # === Plots ===
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Circulatory Failure')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("circulatory_failure_roc.png")

    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.title('Precision-Recall Curve - Circulatory Failure')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig("circulatory_failure_pr.png")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Circulatory Failure")
    plt.savefig("circulatory_failure_cm.png")

if __name__ == "__main__":
    main()
