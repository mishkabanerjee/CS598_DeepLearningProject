# models/evaluate_los_nopretrain.py

import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.los_dataset import HiRIDLOSDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDLOSDataset(
    npy_dir=PretrainConfig.data_dir,
    label_path="data/length_of_stay/los_labels.csv",
    max_len=PretrainConfig.max_len,
)

n = len(dataset)
n_train = int(0.8 * n)
n_test = n - n_train
_, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load model
model = TRACEModel(
    input_dim=PretrainConfig.input_dim,
    hidden_dim=PretrainConfig.hidden_dim,
    num_layers=PretrainConfig.num_layers,
    num_heads=PretrainConfig.num_heads,
    dropout=PretrainConfig.dropout,
).to(device)
regressor = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)

# ✅ Correct checkpoint name
checkpoint = torch.load("ckpt/trace_los_nopretrain.pt", map_location=device)
model.encoder.load_state_dict(checkpoint["encoder"])
regressor.load_state_dict(checkpoint["regressor"])

model.eval()
regressor.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        data = batch["data"].to(device)
        labels = batch["label"].to(device)

        encoded = model.encoder(data)
        pooled = encoded.mean(dim=1)
        preds = regressor(pooled).squeeze(-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# Convert to numpy
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("Prediction shape:", all_preds.shape)
print("Label shape:", all_labels.shape)

# Metrics
mse = mean_squared_error(all_labels, all_preds)
mae = mean_absolute_error(all_labels, all_preds)

print(f"✅ LOS MSE: {mse:.4f}")
print(f"✅ LOS MAE: {mae:.4f}")

# Plot
plt.figure()
plt.scatter(all_labels, all_preds, alpha=0.2)
plt.xlabel("True LOS (days)")
plt.ylabel("Predicted LOS (days)")
plt.title("LOS Prediction (No Pretraining)")
plt.grid()
plt.savefig("los_scatter_nopretrain.png")
plt.show()
