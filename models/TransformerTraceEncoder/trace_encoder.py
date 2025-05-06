import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class TRACEEncoder(nn.Module):
    def __init__(self, input_dim=18, conv_dim=128, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, max_len=256):
        super().__init__()
        
        # 1D Convolution
        self.input_proj = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=1)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(conv_dim, max_len=max_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            src_key_padding_mask: (batch_size, seq_len), True where padding
        """
        # Conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.input_proj(x) # (batch, conv_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_dim)
        
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return output
