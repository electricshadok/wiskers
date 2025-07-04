import pytest
import torch

from wiskers.common.modules.moe import FFN, MoE


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model",
    [
        (2, 16, 8),
        (4, 32, 4),
    ],
)
def test_ffn(batch_size, seq_len, d_model):
    x = torch.randn(batch_size, seq_len, d_model)
    net = FFN(d_model, d_model // 2)
    out_x = net(x)

    assert out_x.shape == (batch_size, seq_len, d_model)
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model",
    [
        (2, 16, 8),
        (4, 32, 4),
    ],
)
def test_moe(batch_size, seq_len, d_model):
    x = torch.randn(batch_size, seq_len, d_model)
    net = MoE(d_model, d_model // 2, num_experts=2)
    out_x = net(x)

    assert out_x.shape == (batch_size, seq_len, d_model)
    assert out_x.dtype == x.dtype
