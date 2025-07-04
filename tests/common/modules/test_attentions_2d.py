import pytest
import torch

from wiskers.common.modules.attentions_2d import SelfMultiheadAttention2D, SelfScaledDotProductAttention2D


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 16, 64, 64),
        (8, 16, 32, 32),
    ],
)
def test_self_multihead_attention(batch_size, in_channels, height, width):
    mha = SelfMultiheadAttention2D(in_channels, num_heads=4)
    x = torch.randn(batch_size, in_channels, height, width)

    out_x = mha(x)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 16, 64, 64),
        (8, 16, 32, 32),
    ],
)
def test_self_scaled_dot_product_attention_2d(batch_size, in_channels, height, width):
    self_attention = SelfScaledDotProductAttention2D(in_channels)
    x = torch.randn(batch_size, in_channels, height, width)

    out_x = self_attention(x)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype
