import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.los_dataset import HiRIDLOSDataset

# === Hyperparameters (local to script) ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
GENERAL_TABLE_PATH = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"
SAVE_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_los.pt"
PRETRAINED_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch50.pt"

WINDOW_SIZE = 12
ENCODING_DIM = 64
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 3  # <=3 days, 4-7 days, >7 days


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = HiRIDLOSDataset(
        npy_dir=BASE_NPY_DIR,
        general_table_path=GENERAL_TABLE_PATH,
        max_len=256
    )

    # Split
    n = len(dataset)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*n), n - int(0.8*n)], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Load encoder and classifier
    encoder = CausalCNNEncoder(
        in_channels=dataset[0]['data'].shape[1],
        out_channels=64,
        depth=4,
        reduced_size=16,
        encoding_size=ENCODING_DIM,
        kernel_size=3,
        window_size=WINDOW_SIZE
    ).to(device)

    encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = nn.Linear(ENCODING_DIM, NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    # Train
    for epoch in range(1, NUM_EPOCHS+1):
        classifier.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            data = batch["data"].to(device)
            labels = batch["label"].to(device).long()

            with torch.no_grad():
                encoded = encoder(data).mean(dim=1)

            logits = classifier(encoded)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {epoch_loss / len(train_loader):.4f}")

    # Save
    torch.save({
        "encoder": encoder.state_dict(),
        "classifier": classifier.state_dict()
    }, SAVE_PATH)
    print("\u2705 Fine-tuned model saved for LOS prediction.")

    # Optional evaluation
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"].to(device)
            labels = batch["label"].cpu()
            encoded = encoder(data).mean(dim=1)
            logits = classifier(encoded).cpu()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
