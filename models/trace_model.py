import torch
import torch.nn as nn
from models.trace_encoder import TRACEEncoder
from utils.masking import random_masking

class TRACEModel(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1, max_len=256):
        super().__init__()
        self.encoder = TRACEEncoder(
            input_dim=input_dim,            
            conv_dim=hidden_dim,
            num_layers=num_layers,
            nhead=num_heads,
            dropout=dropout,
            max_len=max_len,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Predict original features
        )

    def forward(self, x, mask_ratio=0.15):
        # x: (batch_size, seq_len, input_dim)
        
        masked_x, mask = random_masking(x, mask_ratio)
        z = self.encoder(masked_x)  # (batch_size, hidden_dim)
        
        pred = self.projection_head(z)  # (batch_size, input_dim)
        return pred, mask
