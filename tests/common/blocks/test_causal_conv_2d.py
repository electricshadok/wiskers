import pytest
import torch

from wiskers.common.blocks.causal_conv_2d import CausalConv2d


@pytest.mark.parametrize(
    "batch_size, height, width, in_channels",
    [
        (4, 32, 32, 16),
        (4, 64, 64, 8),
    ],
)
def test_causal_conv_2d(batch_size, height, width, in_channels):
    out_channels = 3
    kernel_size = (5, 5)  # Example kernel size
    net = CausalConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    # Note: CausalConv2d expects (N, in_channels, H, W)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, height, width)
    assert out_x.dtype == x.dtype
