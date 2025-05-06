import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from models.TransformerTraceEncoder.trace_model import TRACEModel
from data.los_dataset import HiRIDLOSDataset
from models.TransformerTraceEncoder.train_pretrain_config import PretrainConfig

# ----------------------------
# Setup
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = HiRIDLOSDataset(
    npy_dir=PretrainConfig.data_dir,
    label_path="data/length_of_stay/los_labels.csv",
    max_len=PretrainConfig.max_len,
)

# Train/test split
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

checkpoint = torch.load("ckpt/trace_finetune_los.pt", map_location=device)
model.encoder.load_state_dict(checkpoint["encoder"])
regressor = torch.nn.Linear(PretrainConfig.hidden_dim, 1).to(device)
regressor.load_state_dict(checkpoint["regressor"])

model.encoder.eval()
regressor.eval()

# ----------------------------
# Evaluation
# ----------------------------

all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        data = batch["data"].to(device)
        labels = batch["label"].cpu().numpy()

        encoded = model.encoder(data)
        pooled = encoded.mean(dim=1)
        preds = regressor(pooled).squeeze(-1).cpu().numpy()

        all_labels.extend(labels)
        all_preds.extend(preds)

# Metrics
mae = mean_absolute_error(all_labels, all_preds)
rmse = mean_squared_error(all_labels, all_preds, squared=False)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")

# Plot predicted vs actual
plt.scatter(all_labels, all_preds, alpha=0.5)
plt.xlabel("Actual LOS (days)")
plt.ylabel("Predicted LOS (days)")
plt.title("LOS Prediction")
plt.grid(True)
plt.savefig("los_scatter_plot.png")
plt.show()
