import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Generate time encodings for a batch of timestamps.

    Shapes:
        Input: (N)
        Output: (N, time_dim)
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, t: torch.LongTensor):
        t = t.unsqueeze(-1)  # (B, 1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.time_dim, 2, device=t.device).float() / self.time_dim))
        pos_enc_a = torch.sin(t.repeat(1, self.time_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.time_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
