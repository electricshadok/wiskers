import pytest
import torch

from wiskers.common.modules.causal_conv_1d import CausalConv1d


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim",
    [
        (4, 64, 16),
        (4, 128, 8),
    ],
)
def test_causal_conv_1d(batch_size, seq_len, embed_dim):
    out_channels = 3
    net = CausalConv1d(in_channels=embed_dim, out_channels=out_channels, kernel_size=9)
    # Note: CausalConv1 expects (N, in_channels, seq_length) !
    x = torch.randn(batch_size, embed_dim, seq_len)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, seq_len)
    assert out_x.dtype == x.dtype
