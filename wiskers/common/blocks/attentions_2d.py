import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfScaledDotProductAttention2D(nn.Module):
    """
    Spatial scaled dot-product attention.

    Args:
        channels (int): Number of input channels.

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, x):
        N, C, H, W = x.size()

        # Attention mechanism expects a tensor of shape (B, seq_len, C)
        # N is the batch_size
        # seq_len is treated as the product of H and W
        # C is the input_dim
        x = x.view(N, C, -1).swapaxes(1, 2)
        # LayerNorm armonizes per feature map independently
        x_ln = self.ln(x)

        q = self.query(x_ln)
        k = self.key(x_ln)
        v = self.value(x_ln)

        scores = torch.bmm(q, k.transpose(1, 2))
        scores /= self.channels**0.5

        attention = F.softmax(scores, dim=-1)
        weighted_v = torch.bmm(attention, v)

        # Reshape the shape to its original
        att_x = weighted_v.swapaxes(1, 2).view(N, C, H, W)
        return att_x


class SelfMultiheadAttention2D(nn.Module):
    """Multihead attention module

    Args:
        channels (int): Number of input channels.
        num_heads (int): Number of parallel attention heads.

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("embed_dim muse be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Project input features to channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        # Attention mechanism expects a tensor of shape (B, seq_len, C)
        # N is the batch_size
        # seq_len is treated as the product of H and W
        # C is the input_dim
        N, C, H, W = x.size()

        # Flatten the spatial dimensions into [N, H*W, C]
        x = x.view(N, C, H * W).transpose(1, 2)

        # Linear projections into [N, H*W, num_heads, head_dim]
        q = self.query(x).view(N, -1, self.num_heads, self.head_dim)
        k = self.key(x).view(N, -1, self.num_heads, self.head_dim)
        v = self.value(x).view(N, -1, self.num_heads, self.head_dim)

        # Reshape into [N, num_heads, H*W, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= self.head_dim**0.5
        attention = F.softmax(scores, dim=-1)
        weighted_v = torch.matmul(attention, v)

        # Concatenate heads and put through final linear layer
        weighted_v = weighted_v.transpose(1, 2).contiguous().view(N, H * W, self.channels)

        # Reshape back to the original 2D shape [N, channels, H, W]
        output = self.out(weighted_v).view(N, H, W, self.channels).transpose(1, 3)

        return output
