import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add root to sys.path to import modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.causal_cnn_encoder import CausalCNNEncoder
from data.contrastive_dataset import ContrastiveACFDataset

# === Hyperparameters (local to script) ===
BASE_NPY_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
ACF_NEIGHBOR_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"
SAVE_DIR = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt"
WINDOW_SIZE = 12
ENCODING_DIM = 10  # Match encoding_size with model definition
BATCH_SIZE = 64
NUM_EPOCHS = 25  # Reduced from 50 to 25 due to computing constraints

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, other, label):
        dists = torch.norm(anchor - other, dim=1)
        loss = label * dists.pow(2) + (1 - label) * torch.clamp(self.margin - dists, min=0).pow(2)
        return loss.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ContrastiveACFDataset(BASE_NPY_DIR, ACF_NEIGHBOR_DIR, window_size=WINDOW_SIZE)
    if len(dataset) == 0:
        print("⚠️ No usable contrastive samples found.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Use model config similar to TRACE
    model = CausalCNNEncoder(
        in_channels=dataset[0]["anchor"].shape[1],
        out_channels=64,
        depth=4,
        reduced_size=4,  # Ensures correct flattening
        encoding_size=ENCODING_DIM,  # Match fine-tuning
        kernel_size=3,
        window_size=WINDOW_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss(margin=1.0)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            anchor = batch["anchor"].to(device)
            other = batch["other"].to(device)
            label = batch["label"].to(device)

            encoded_anchor = model(anchor)
            encoded_other = model(other)

            loss = criterion(encoded_anchor, encoded_other, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        # Save model every 5 epochs
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"causalcnn_pretrain_epoch{epoch}.pt"))
            print(f"✅ Saved model at epoch {epoch}")

if __name__ == "__main__":
    main()
