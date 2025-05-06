import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.RandomEncoder.random_encoder import RandomCausalConvEncoder

# === Imports from project ===
from config import (
    BASE_NPY_DIR, MASK_SUFFIX, LABEL_PATHS,
    ENCODING_OUTPUT_DIR, WINDOW_SIZE, ENCODING_DIM,
    BATCH_SIZE, SEED
)

# === Task Selection ===
TASK = "circulatory_failure"

# === Prepare label and patient ID data ===
TRAIN_IDS = np.load(LABEL_PATHS[TASK]["train_ids"])
TRAIN_LABELS = np.load(LABEL_PATHS[TASK]["train_labels"])
TEST_IDS = np.load(LABEL_PATHS[TASK]["test_ids"])
TEST_LABELS = np.load(LABEL_PATHS[TASK]["test_labels"])

# Build a combined label dictionary
labels_dict = {int(pid): int(label) for pid, label in zip(TRAIN_IDS, TRAIN_LABELS)}
labels_dict.update({int(pid): int(label) for pid, label in zip(TEST_IDS, TEST_LABELS)})

# === Dataset Definition ===
class RandomEncodingDataset(Dataset):
    def __init__(self, patient_ids, labels_dict):
        self.ids = patient_ids
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = int(self.ids[idx])
        x = np.load(os.path.join(BASE_NPY_DIR, f"patient_{pid}.npy"))
        mask = np.load(os.path.join(BASE_NPY_DIR, f"patient_{pid}{MASK_SUFFIX}"))

        x_tensor = torch.tensor(x, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        label = int(self.labels_dict[pid])
        return x_tensor, mask_tensor, label

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    xs, masks, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True)  # Pads along T dimension
    masks_padded = pad_sequence(masks, batch_first=True)
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return xs_padded, masks_padded, ys_tensor


train_loader = DataLoader(
    RandomEncodingDataset(TRAIN_IDS, labels_dict),batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    RandomEncodingDataset(TEST_IDS, labels_dict),batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# === Random Encoder ===
encoder = RandomCausalConvEncoder(input_dim=18, hidden_dim=64, encoding_dim=ENCODING_DIM)
encoder.eval()

def encode_dataset(dataloader):
    features, targets = [], []
    with torch.no_grad():
        for x, mask, y in tqdm(dataloader, desc="Encoding"):
            z = encoder(x)
            z_pooled = torch.mean(z, dim=1)  # Global average pooling
            features.append(z_pooled.cpu().numpy())
            targets.extend(y.numpy())
    return np.vstack(features), np.array(targets)


# === Encoding and Classification ===
X_train, y_train = encode_dataset(train_loader)
X_test, y_test = encode_dataset(test_loader)

# === Save encoded features ===
os.makedirs(ENCODING_OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(ENCODING_OUTPUT_DIR, f"{TASK}_X_train_random.npy"), X_train)
np.save(os.path.join(ENCODING_OUTPUT_DIR, f"{TASK}_y_train.npy"), y_train)
np.save(os.path.join(ENCODING_OUTPUT_DIR, f"{TASK}_X_test_random.npy"), X_test)
np.save(os.path.join(ENCODING_OUTPUT_DIR, f"{TASK}_y_test.npy"), y_test)
print(f"💾 Saved encoded features for {TASK} to {ENCODING_OUTPUT_DIR}")

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]

auroc = roc_auc_score(y_test, y_pred)
print(f"🎯 Random Encoder AUROC ({TASK}): {auroc:.4f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Threshold of 0.5 to get binary predictions
y_pred_binary = (y_pred >= 0.5).astype(int)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap="Blues")
plt.title(f"{TASK.capitalize()} – Random Encoder Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{TASK}_confusion_matrix_random.png")
plt.show()

