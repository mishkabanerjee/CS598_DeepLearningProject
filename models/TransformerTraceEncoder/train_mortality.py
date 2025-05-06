# ============================
# train_mortality.py (FULL)
# ============================

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.mortality_dataset import HiRIDMortalityDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# -----------------------------
# Setup
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
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

# Load pretrained encoder
pretrained_model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)

checkpoint_path = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\trace_pretrain_epoch50.pt"
pretrained_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Freeze encoder
for param in pretrained_model.parameters():
    param.requires_grad = False

encoder = pretrained_model.encoder
encoder.eval()

# Mortality classifier
classifier = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=PretrainConfig.learning_rate)
loss_fn = torch.nn.BCEWithLogitsLoss()

# -----------------------------
# Fine-tuning loop
# -----------------------------

for epoch in range(1, PretrainConfig.num_epochs + 1):
    classifier.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            encoded = encoder(data)  # (B, T, H)
            pooled = encoded.mean(dim=1)  # (B, H)

        logits = classifier(pooled).squeeze(-1)  # (B,)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{PretrainConfig.num_epochs}] - Fine-tuning Loss: {avg_loss:.6f}")

# Save both
os.makedirs("ckpt", exist_ok=True)
torch.save({
    "encoder": encoder.state_dict(),
    "classifier": classifier.state_dict()
}, "ckpt/trace_finetune_mortality.pt")
print("\u2705 Fine-tuning complete.")
