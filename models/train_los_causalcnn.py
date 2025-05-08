import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.los_dataset_clean import HiRIDLOSDatasetClean

# === Hyperparameters ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
LABEL_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\length_of_stay\los_labels.csv"
PRETRAINED_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_pretrain_epoch25.pt"
SAVE_PATH = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\causalcnn_finetune_los.pt"

WINDOW_SIZE = 12
ENCODING_DIM = 10
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HiRIDLOSDatasetClean(BASE_NPY_DIR, LABEL_PATH, max_len=WINDOW_SIZE)
    print(f"\u2705 Loaded {len(dataset)} LOS samples.")

    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
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

    encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = nn.Linear(ENCODING_DIM, 3).to(device)  # 3 LOS classes
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        classifier.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            data = batch["data"].to(device)
            labels = batch["label"].long().to(device)

            with torch.no_grad():
                encoded = encoder(data)

            logits = classifier(encoded)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, SAVE_PATH)
    print("\u2705 Fine-tuned model saved for LOS prediction.")

    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"].to(device)
            labels = batch["label"].cpu()
            encoded = encoder(data)
            preds = torch.softmax(classifier(encoded), dim=1).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ LOS Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
