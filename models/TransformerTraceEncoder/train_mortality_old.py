import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.utils.data import DataLoader
from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.mortality_dataset import HiRIDMortalityDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt\trace_pretrain_epoch50.pt"  # <-- your pretrained checkpoint path
npy_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\npy"
general_table_path = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"

# -----------------------------
# Load pretrained encoder
# -----------------------------
pretrained_model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)
pretrained_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
pretrained_model.eval()

# Freeze encoder weights (optional: you can also allow fine-tuning later)
for param in pretrained_model.parameters():
    param.requires_grad = False

# -----------------------------
# Build classifier head
# -----------------------------
classifier = nn.Sequential(
    nn.Linear(PretrainConfig.hidden_dim, 1),
).to(device)

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

# -----------------------------
# Load mortality dataset
# -----------------------------
dataset = HiRIDMortalityDataset(
    npy_dir=npy_dir,
    general_table_path=general_table_path,
    max_len=PretrainConfig.max_len,
)
dataloader = DataLoader(
    dataset,
    batch_size=PretrainConfig.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

# -----------------------------
# Fine-tuning loop
# -----------------------------
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    classifier.train()
    total_loss = 0

    for batch in dataloader:
        data = batch["data"].to(device)  # (B, T, D)
        time = batch["time"].to(device)  # (B, T)
        mask = batch["mask"].to(device)  # (B, T, D)
        labels = batch["label"].to(device)  # (B,)

        with torch.no_grad():
            encoded = pretrained_model.encoder(data)

            # Pool over time
            encoded = encoded.mean(dim=1)  # (B, hidden_dim)

        preds = classifier(encoded).squeeze()  # (B,)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch}/{num_epochs}] - Fine-tuning Loss: {avg_loss:.6f}")

print("✅ Fine-tuning complete.")


# Save fine-tuned model
os.makedirs("ckpt", exist_ok=True)
torch.save(pretrained_model.state_dict(), "ckpt/trace_finetune_mortality.pt")
print("✅ Fine-tuned model saved to ckpt/trace_finetune_mortality.pt")
