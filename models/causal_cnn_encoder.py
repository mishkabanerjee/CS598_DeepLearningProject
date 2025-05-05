import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 36,
        out_channels: int = 4,
        depth: int = 1,
        reduced_size: int = 2,
        encoding_size: int = 10,
        kernel_size: int = 2,
        window_size: int = 12
    ):
        super(CausalCNNEncoder, self).__init__()

        self.window_size = window_size
        layers = []
        dilation = 1

        for i in range(depth):
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            dilation *= 2

        self.network = nn.Sequential(*layers)

        self.reduction_layer = nn.AdaptiveAvgPool1d(reduced_size)
        self.output_layer = nn.Linear(out_channels * reduced_size, encoding_size)

    def forward(self, x):
        # x shape: (batch_size, window_size, input_channels)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        out = self.network(x)
        out = self.reduction_layer(out)  # (B, C, reduced_size)
        out = out.view(out.size(0), -1)  # flatten
        encoding = self.output_layer(out)  # (B, encoding_size)
        return encoding
