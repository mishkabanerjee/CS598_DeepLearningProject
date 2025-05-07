import sys
import os
sys.path.append(os.getcwd())

import torch
from models.trace_model import TRACEModel
from models.trace_encoder import TRACEEncoder
from data.hirid_dataset import HiRIDDataset
from utils.masking import random_masking
from models.train_pretrain_config import PretrainConfig
from torch.utils.data import DataLoader

# -----------------------------
# Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = HiRIDDataset(
    data_dir=PretrainConfig.data_dir,
    max_len=PretrainConfig.max_len,
)

dataloader = DataLoader(
    dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=PretrainConfig.learning_rate)
loss_fn = torch.nn.MSELoss()

# -----------------------------
# Training Loop
# -----------------------------

train_losses = []   # <== ADD THIS before the epoch loop

for epoch in range(1, PretrainConfig.num_epochs + 1):
    model.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)      # (B, T, D)
        time = batch["time"].to(device)       # (B, T)
        mask = batch["mask"].to(device)       # (B, T, D)

        # Apply random masking        
        masked_data, mask_indicator = random_masking(data, mask_ratio=PretrainConfig.mask_ratio)


        # Forward pass
        pred, _ = model(masked_data)

        # Only compute loss on masked positions
        loss = loss_fn(pred[mask_indicator], data[mask_indicator])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{PretrainConfig.num_epochs}] - Loss: {avg_loss:.6f}")

    train_losses.append(avg_loss)


    # Save checkpoint
    if epoch % PretrainConfig.save_every == 0:
        os.makedirs(PretrainConfig.save_dir, exist_ok=True)
        save_path = os.path.join(PretrainConfig.save_dir, f"trace_pretrain_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved checkpoint: {save_path}")

import matplotlib.pyplot as plt

plt.plot(range(1, PretrainConfig.num_epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Pretraining Loss Curve')
plt.grid(True)
plt.show()

