import pytest
import torch

from wiskers.common.blocks.causal_conv import (
    CausalConv1d,
    CausalConv2d,
    CausalConv3d,
)


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
    x = torch.randn(batch_size, embed_dim, seq_len)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, seq_len)
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, height, width, in_channels",
    [
        (4, 32, 32, 16),
        (4, 64, 64, 8),
    ],
)
def test_causal_conv_2d(batch_size, height, width, in_channels):
    out_channels = 3
    kernel_size = (5, 5)
    net = CausalConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, height, width)
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, depth, height, width, in_channels",
    [
        (4, 16, 32, 32, 8),
        (2, 32, 64, 64, 16),
    ],
)
def test_causal_conv_3d(batch_size, depth, height, width, in_channels):
    out_channels = 3
    kernel_size = (3, 3, 3)
    net = CausalConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, depth, height, width)
    assert out_x.dtype == x.dtype

