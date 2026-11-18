import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Generate time encodings for a batch of timestamps.
    Usage: add to input embeddings
    Positional type: absolute

    Shapes:
        Input: (N)
        Output: (N, time_dim)
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, t: torch.LongTensor):
        t = t.unsqueeze(-1)  # (B, 1)
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.time_dim, 2, device=t.device).float()
                / self.time_dim
            )
        )
        pos_enc_a = torch.sin(t.repeat(1, self.time_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.time_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


class RoPE(nn.Module):
    """
    Implementation of Rotary Positional Embedding (RoPE)
    Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding."
    - Usage: RoPE encodes position inside the attention operation itself (by rotating query/key features).
    It gives relative position awareness without adding embeddings to inputs.
    - Positional type: relative
    - Intuition: tell the attention how far apart two tokens are.

    Args:
        dim_head (int): Dimensionality of each attention head (must be even).
        max_seq_len (int): Maximum sequence length for which to precompute sin/cos tables (default: 1024).
        base (float): Base frequency scaling factor controlling rotation rate (default: 10000.0).

    Reference:
        https://huggingface.co/microsoft/Phi-3-small-8k-instruct/blob/main/positional_embedding.py
    """

    def __init__(self, dim_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()

        self.dim_head = dim_head
        self.max_seq_len = max_seq_len
        self.base = base

        # 1. build frequency spectrum
        half_dim = dim_head // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim)
        )

        # 2. create positional indices [0, 1, 2, ..., max_seq_len - 1]
        t = torch.arange(max_seq_len, dtype=torch.float32)

        # 3. outer product to create phase matrix [pos, dim]
        freqs = torch.outer(t, inv_freq)

        # 4. compute rotation coefficients
        cos, sin = freqs.cos(), freqs.sin()

        # 5. duplicate to full dim_head (real, imaginary halves)
        self.register_buffer(
            "cos", torch.stack([cos, cos], dim=-1).reshape(max_seq_len, dim_head)
        )
        self.register_buffer(
            "sin", torch.stack([sin, sin], dim=-1).reshape(max_seq_len, dim_head)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, num_head, dim_head) (queries or keys).

        Returns:
            torch.Tensor: Rotated tensor of same shape (B, seq_len, num_head, dim_head).
        """
        B, seq_len, num_head, dim_head = x.shape

        # slice the precomputed lookup tables
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)  # (1, seq_len, 1, dim_head)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)  # (1, seq_len, 1, dim_head)

        # Apply 2D rotation to each (even, odd) feature pair in x.
        #
        # Example for a single 2D vector x = [x1, x2]:
        #   x1' = x1 * cos(a) - x2 * sin(a)
        #   x2' = x1 * sin(a) + x2 * cos(a)
        #
        # For a full feature vector x = [x1, x2, x3, ..., xD],
        # we apply the same rotation pattern to every (x2k, x2k+1) pair.
        #
        # The compact vectorized form is:
        #   x_rot = (x * cos) + (rotate_half(x) * sin)
        #
        # Here, rotate_half(x) use the "half-split" convention (real and imaginary halves):
        #   rotate_half(x) = [-x_imag, x_real]

        x1, x2 = x[..., : dim_head // 2], x[..., dim_head // 2 :]
        rotate_half = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotate_half * sin)
