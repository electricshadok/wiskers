import pytest
import torch

from wiskers.common.blocks.transformer_2d import TransformerSelfAttention2D


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 16, 64, 64),
        (8, 16, 32, 32),
    ],
)
def test_transformer_attention_2d(batch_size, in_channels, height, width):
    # Note: in_channels must be divisible by num_heads
    self_attention = TransformerSelfAttention2D(in_channels, num_heads=8)
    x = torch.randn(batch_size, in_channels, height, width)

    out_x = self_attention(x)

    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype
