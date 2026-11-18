import torch.nn as nn


class TransformerSelfAttention2D(nn.Module):
    """
    Self-attention module with multi-head attention and feed-forward layers.

    Args:
        channels (int): Input and output channel count.
        num_heads (int): Number of attention heads.

    Notes:
        This module cannot be exported to ONNX
        'aten::_native_multi_head_attention' to ONNX opset version 14 is not supported

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, channels, num_heads):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # in: [N, C, H, W]
        # out: [N, C, H, W]
        N, C, H, W = x.size()
        # MultiheadAttention expect [batch_size, seq_len, embed_dim]
        # N is the batch_size
        # seq_len is treated as the product of H and W
        # C is the embed_dim
        x = x.view(N, C, -1).swapaxes(1, 2)
        # LayerNorm armonizes per feature map independently
        x_ln = self.ln(x)
        # Add multihead attention
        att_x, _ = self.mha(x_ln, x_ln, x_ln)
        att_x = att_x + x
        att_x = self.ff(att_x) + att_x
        # Reshape the shape to its original
        att_x = att_x.swapaxes(1, 2).view(N, C, H, W)

        return att_x
