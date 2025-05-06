import torch
import torch.nn as nn

class RandomCausalConvEncoder(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, encoding_dim=64, kernel_size=3, num_layers=4):
        super().__init__()
        layers = []

        # First layer
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size - 1))
        layers.append(nn.ReLU())

        # Dilated causal conv layers
        for i in range(1, num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation
            ))
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)
        self.final = nn.Conv1d(hidden_dim, encoding_dim, 1)

        # Freeze all weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, mask=None):
        """
        x: (B, T, D)
        Returns: (B, T, encoding_dim)
        """
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)
        z = self.final(x)
        return z.transpose(1, 2)
