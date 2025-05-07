import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from models.trace_model import TRACEModel
from data.los_dataset import HiRIDLOSDataset
from models.train_pretrain_config import PretrainConfig

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDLOSDataset(
    npy_dir=PretrainConfig.data_dir,
    label_path="data/los_labels.csv",
    max_len=PretrainConfig.max_len,
)

dataloader = DataLoader(
    dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# Load pretrained TRACE model
pretrained_model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)

checkpoint_path = os.path.join(PretrainConfig.save_dir, "trace_pretrain_epoch50.pt")
pretrained_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Freeze encoder
for param in pretrained_model.parameters():
    param.requires_grad = False

encoder = pretrained_model.encoder
encoder.eval()

# Linear regression head
regressor = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)
optimizer = torch.optim.Adam(regressor.parameters(), lr=PretrainConfig.learning_rate)
loss_fn = torch.nn.MSELoss()

# -------------------------
# Training loop
# -------------------------
for epoch in range(1, PretrainConfig.num_epochs + 1):
    regressor.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)  # shape: (B,)

        with torch.no_grad():
            encoded = encoder(data)
            pooled = encoded.mean(dim=1)  # shape: (B, H)

        preds = regressor(pooled).squeeze(-1)  # shape: (B,)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{PretrainConfig.num_epochs}] - LOS MSE Loss: {avg_loss:.6f}")

# -------------------------
# Save checkpoint
# -------------------------
os.makedirs("ckpt", exist_ok=True)
torch.save({
    "encoder": encoder.state_dict(),
    "regressor": regressor.state_dict()
}, "ckpt/trace_finetune_los.pt")

print("✅ LOS fine-tuning complete.")
