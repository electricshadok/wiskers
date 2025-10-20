import pytest
import torch

from wiskers.common.blocks.positional_encoding import RoPE, SinusoidalPositionEmbedding


@pytest.mark.parametrize("batch_size, time_dim", [(16, 128), (64, 512)])
def test_time_encoding(batch_size, time_dim):
    step_id = 12
    t = (torch.ones(batch_size) * step_id).long()  # (B)

    pe = SinusoidalPositionEmbedding(time_dim)
    te = pe(t)  # (B, time_dim)

    assert te.shape == (batch_size, time_dim)
    assert te.dtype == torch.float32


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, dim_head", [(2, 16, 8, 128), (4, 32, 4, 64)]
)
def test_rope_encoding(batch_size, seq_len, num_heads, dim_head):
    x = torch.randn(batch_size, seq_len, num_heads, dim_head)

    rope = RoPE(dim_head=dim_head, max_seq_len=seq_len)
    x_rot = rope(x)

    assert x_rot.shape == x.shape
    assert x_rot.dtype == torch.float32
