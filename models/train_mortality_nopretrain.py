import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from data.mortality_dataset import HiRIDMortalityDataset
from models.trace_model import TRACEModel
from models.train_pretrain_config import PretrainConfig


# --- Override number of epochs for quick test ---
PretrainConfig.num_epochs = 300

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDMortalityDataset(
    npy_dir=PretrainConfig.data_dir,
    general_table_path=r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv",
    max_len=PretrainConfig.max_len,
)

dataloader = DataLoader(
    dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# Initialize TRACE encoder from scratch
encoder = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).encoder.to(device)

# Classification head
classifier = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=PretrainConfig.learning_rate)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(1, PretrainConfig.num_epochs + 1):
    encoder.train()
    classifier.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        encoded = encoder(data)             # (B, T, H)
        pooled = encoded.mean(dim=1)        # (B, H)
        logits = classifier(pooled).squeeze(-1)  # (B,)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{PretrainConfig.num_epochs}] - Mortality Loss (No Pretrain): {avg_loss:.6f}")

# Save checkpoint
os.makedirs("ckpt", exist_ok=True)
torch.save({
    "encoder": encoder.state_dict(),
    "classifier": classifier.state_dict()
}, "ckpt/trace_finetune_mortality_nopretrain.pt")
print("✅ Mortality classification (no pretrain) complete.")



