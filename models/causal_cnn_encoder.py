import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels=18,
        out_channels=64,
        depth=6,
        reduced_size=16,
        encoding_size=10,
        kernel_size=3,
        window_size=12
    ):
        super().__init__()
        self.depth = depth
        self.reduced_size = reduced_size

        layers = []
        for i in range(depth):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                )
            )
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.final = nn.Linear(out_channels * window_size, encoding_size)

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        out = self.network(x)[:, :, -x.size(-1):]  # Ensure causality
        out = out.reshape(out.size(0), -1)  # (B, D*T)
        out = self.final(out)  # (B, encoding_size)
        return out
