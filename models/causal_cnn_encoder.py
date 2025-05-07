import torch
import torch.nn as nn

class CausalCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels=18,
        out_channels=64,
        depth=4,
        reduced_size=4,         # Unused here, but keeping for future use
        encoding_size=10,       # Must match classifier input
        kernel_size=3,
        window_size=12
    ):
        super().__init__()
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
        x = x.permute(0, 2, 1)  # (B, D, T)
        out = self.network(x)[:, :, -x.size(-1):]  # (B, C, T)
        out = out.reshape(out.size(0), -1)  # (B, C*T) = (B, 64*12)
        return self.final(out)  # (B, encoding_size)
