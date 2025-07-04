import pytest
import torch

from wiskers.common.modules.attentions_1d import MultiheadAttention, ScaledDotProductAttention


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim",
    [
        (4, 512, 64),
        (8, 1024, 32),
    ],
)
def test_scaled_dot_product_attention(batch_size, seq_len, embed_dim):
    self_attention = ScaledDotProductAttention(embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out_x = self_attention(x, x, x)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim",
    [
        (4, 512, 64),
        (8, 1024, 32),
    ],
)
def test_multihead_attention(batch_size, seq_len, embed_dim):
    attention = MultiheadAttention(embed_dim=embed_dim, num_heads=embed_dim // 2)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out_x = attention(x, x, x)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype
