# models/train_los_nopretrain.py

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.los_dataset import HiRIDLOSDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# --- Override number of epochs for quick test ---
PretrainConfig.num_epochs = 50

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

# TRACE model (no pretrained weights)
model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)

# Regressor head
regressor = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=PretrainConfig.learning_rate)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(1, PretrainConfig.num_epochs + 1):
    model.train()
    regressor.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        encoded = model.encoder(data)
        pooled = encoded.mean(dim=1)

        logits = regressor(pooled).squeeze(-1)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{PretrainConfig.num_epochs}] - LOS MSE Loss: {avg_loss:.6f}")

# Save checkpoint (✅ this format supports evaluation)
os.makedirs("ckpt", exist_ok=True)
torch.save({
    "encoder": model.encoder.state_dict(),
    "regressor": regressor.state_dict()
}, "ckpt/trace_los_nopretrain.pt")

print("✅ LOS no-pretrain training complete (50-epoch test run).")
