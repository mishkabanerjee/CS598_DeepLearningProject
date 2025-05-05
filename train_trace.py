import os
import torch
from torch.utils.data import DataLoader
from datasets.hirid_window_dataset import HiRIDWindowDataset
from models.causal_cnn_encoder import CausalCNNEncoder
from losses.contrastive_loss import ContrastiveLoss
from torch.optim import Adam
from tqdm import tqdm

# ===========================
# CONFIG
# ===========================
DATA_DIR = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
ACF_DIR  = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"
SAVE_DIR = r"ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 30
WINDOW_SIZE = 12
NUM_NEGATIVES = 5
EPOCHS = 150
LR = 5e-5
WEIGHT_DECAY = 5e-4
TEMPERATURE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# SETUP
# ===========================
print("📦 Loading dataset...")
dataset = HiRIDWindowDataset(DATA_DIR, ACF_DIR, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print("🧠 Building model...")
encoder = CausalCNNEncoder(
    in_channels=dataset[0][0].shape[1],  # D
    out_channels=4,
    depth=1,
    reduced_size=2,
    encoding_size=10,
    kernel_size=2,
    window_size=WINDOW_SIZE
).to(DEVICE)

loss_fn = ContrastiveLoss(temperature=TEMPERATURE)
optimizer = Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ===========================
# TRAIN
# ===========================
print("🚀 Starting training...")
for epoch in range(1, EPOCHS + 1):
    encoder.train()
    total_loss = 0
    for x_anchor, x_pos, x_neg in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x_anchor = x_anchor.to(DEVICE)        # (B, T, D)
        x_pos    = x_pos.to(DEVICE)           # (B, T, D)
        x_neg    = x_neg.to(DEVICE)           # (B, K, T, D)

        # Encode
        z_anchor = encoder(x_anchor)          # (B, 10)
        z_pos    = encoder(x_pos)             # (B, 10)

        B, K, T, D = x_neg.shape
        z_neg = encoder(x_neg.view(B * K, T, D))  # (B*K, 10)
        z_neg = z_neg.view(B, K, -1)              # (B, K, 10)

        # Contrastive loss
        loss = loss_fn(z_anchor, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📉 Epoch {epoch}: avg loss = {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_path = os.path.join(SAVE_DIR, f"encoder_epoch{epoch}.pt")
        torch.save(encoder.state_dict(), save_path)
        print(f"💾 Saved checkpoint: {save_path}")
