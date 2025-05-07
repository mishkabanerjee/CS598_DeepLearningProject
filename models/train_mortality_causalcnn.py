import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.mortality_dataset import HiRIDMortalityDataset

# === Hyperparameters ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
SAVE_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_mortality.pt"
PRETRAINED_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch25.pt"

WINDOW_SIZE = 12
ENCODING_DIM = 10
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HiRIDMortalityDataset(
        npy_dir=BASE_NPY_DIR,
        general_table_path=GENERAL_TABLE_PATH,
        max_len=WINDOW_SIZE
    )

    n = len(dataset)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * n), n - int(0.8 * n)], generator=torch.Generator().manual_seed(42))
    np.save("split_test_indices.npy", test_set.indices)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    encoder = CausalCNNEncoder(
        in_channels=dataset[0]['data'].shape[1],
        out_channels=64,
        depth=4,
        reduced_size=4,
        encoding_size=ENCODING_DIM,
        kernel_size=3,
        window_size=WINDOW_SIZE
    ).to(device)

    encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

    for param in encoder.parameters():
        param.requires_grad = False   

    # Classifier
    classifier = nn.Linear(ENCODING_DIM, 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(1, NUM_EPOCHS + 1):
        classifier.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            data = batch["data"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                encoded = encoder(data)

            logits = classifier(encoded).squeeze()
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, SAVE_PATH)
    print("\u2705 Fine-tuned model saved for mortality prediction.")

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

    print(f"\n\u2705 AUROC: {auroc:.4f}")
    print(f"\u2705 Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
