import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.circulatory_dataset import CirculatoryFailureDataset

# --- Config ---
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABEL_CSV = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\circulatory_labels.csv"
CHECKPOINT_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch50.pt"
WINDOW_SIZE = 12
ENCODING_DIM = 64
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CirculatoryFailureDataset(npy_dir=NPY_DIR, label_csv_path=LABEL_CSV, window_size=WINDOW_SIZE)
n = len(dataset)
n_train = int(0.8 * n)
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

encoder = CausalCNNEncoder(
    in_channels=dataset[0]["data"].shape[1],
    out_channels=64,
    depth=4,
    reduced_size=16,
    encoding_size=ENCODING_DIM,
    kernel_size=3,
    window_size=WINDOW_SIZE
).to(device)

encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
encoder.eval()

classifier = torch.nn.Linear(ENCODING_DIM, 1).to(device)
classifier.eval()

# --- Evaluation ---
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating Circulatory Failure"):
        data = batch["data"].to(device)
        labels = batch["label"].to(device)
        
        features = encoder(data)
        logits = classifier(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

all_labels = torch.cat(all_labels).numpy()
all_probs = torch.cat(all_probs).numpy()

auroc = roc_auc_score(all_labels, all_probs)
accuracy = accuracy_score(all_labels, (all_probs > 0.5).astype(float))

print(f"AUROC: {auroc:.4f}\nAccuracy: {accuracy:.4f}")

fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC - Circulatory Failure")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.savefig("circulatory_roc_causalcnn.png")
plt.show()