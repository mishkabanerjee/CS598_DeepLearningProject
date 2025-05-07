import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Ensures plots save properly without GUI
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.mortality_dataset import HiRIDMortalityDataset

# === Paths ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
MODEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_mortality.pt"
TEST_INDEX_PATH = "split_test_indices.npy"

WINDOW_SIZE = 12
ENCODING_DIM = 10
BATCH_SIZE = 64

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = HiRIDMortalityDataset(
        npy_dir=BASE_NPY_DIR,
        general_table_path=GENERAL_TABLE_PATH,
        max_len=WINDOW_SIZE
    )

    # Load test indices
    if not os.path.exists(TEST_INDEX_PATH):
        raise FileNotFoundError(f"Test index file not found at {TEST_INDEX_PATH}. Run training first.")
    test_indices = np.load(TEST_INDEX_PATH)
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Load model
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
        for batch in test_loader:
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
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, (all_probs > 0.5).astype(int))

    print(f"\n✅ Evaluation Results on Saved Test Set:")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")

    # === Plots ===
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve - Mortality (CausalCNN)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("causalcnn_mortality_roc.png")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.title("Precision-Recall Curve - Mortality (CausalCNN)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("causalcnn_mortality_pr.png")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Mortality (CausalCNN)")
    plt.savefig("causalcnn_mortality_cm.png")
    plt.close()

if __name__ == "__main__":
    main()
