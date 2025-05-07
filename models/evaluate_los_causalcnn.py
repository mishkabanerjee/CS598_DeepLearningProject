mport os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.los_dataset import LengthOfStayDataset

# --- Config ---
NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABEL_CSV = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\los_labels.csv"
CHECKPOINT_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch50.pt"
WINDOW_SIZE = 12
ENCODING_DIM = 64
BATCH_SIZE = 64
NUM_CLASSES = 3  # e.g., short, medium, long stay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LengthOfStayDataset(npy_dir=NPY_DIR, label_csv_path=LABEL_CSV, window_size=WINDOW_SIZE)
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

classifier = torch.nn.Linear(ENCODING_DIM, NUM_CLASSES).to(device)
classifier.eval()

# --- Evaluation ---
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating LOS"):
        data = batch["data"].to(device)
        labels = batch["label"].to(device)
        
        features = encoder(data)
        logits = classifier(features)
        preds = torch.argmax(logits, dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())

all_labels = torch.cat(all_labels).numpy()
all_preds = torch.cat(all_preds).numpy()

print("Classification Report:")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

plt.figure()
plt.hist(all_labels, bins=NUM_CLASSES, alpha=0.5, label='True')
plt.hist(all_preds, bins=NUM_CLASSES, alpha=0.5, label='Predicted')
plt.legend()
plt.title("LOS Prediction Distribution")
plt.savefig("los_distribution_causalcnn.png")
plt.show()
