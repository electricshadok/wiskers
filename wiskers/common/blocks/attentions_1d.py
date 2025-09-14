import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.
    Paper: "Attention is all you need", 2017.

    Args:
        embed_dim (int): Embedding dimension of the sequence.

    Shapes:
        Input: (N, L, E)
        Output: (N, L, E)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0 / self.embed_dim**0.5

        # Linear projections for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        proj_q = self.query(q)
        proj_k = self.key(k)
        proj_v = self.value(v)

        scores = torch.bmm(proj_q, proj_k.transpose(1, 2))
        scores *= self.scale

        attention = self.softmax(scores)
        scaled_values = torch.bmm(attention, proj_v)

        return scaled_values


class MultiheadAttention(nn.Module):
    """
    Multihead attention module.
    Paper: "Attention is all you need", 2017.

    Args:
        embed_dim (int): Embedding dimension of the sequence.
        num_heads (int): Number of attention heads.

    Shapes:
        Input: (N, L, E)
        Output: (N, L, E)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim muse be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0 / self.head_dim**0.5

        # Linear projections for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # MultiheadAttention expects [batch_size, seq_len, embed_dim]
        B, L, _ = q.size()

        # Project and reshape the queries, keys, and values
        proj_q = self.query(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        proj_k = self.key(k).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        proj_v = self.value(v).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(proj_q, proj_k.transpose(-2, -1))  # (B, num_heads, L, L)
        scores *= self.scale
        attention = self.softmax(scores)

        scaled_values = torch.matmul(attention, proj_v)  # (B, num_heads, L, head_dim)

        concat_values = scaled_values.transpose(1, 2).contiguous().view(B, L, self.embed_dim)  # (B, L, embed_dim)
        return self.out(concat_values)
