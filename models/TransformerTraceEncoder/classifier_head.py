import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=2, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, output_dim)
        """
        return self.net(x)
