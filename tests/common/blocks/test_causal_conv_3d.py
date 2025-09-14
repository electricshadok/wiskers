import pytest
import torch

from wiskers.common.blocks.causal_conv_3d import CausalConv3d


@pytest.mark.parametrize(
    "batch_size, depth, height, width, in_channels",
    [
        (4, 16, 32, 32, 8),
        (2, 32, 64, 64, 16),
    ],
)
def test_causal_conv_3d(batch_size, depth, height, width, in_channels):
    out_channels = 3
    kernel_size = (3, 3, 3)  # Example 3D kernel
    net = CausalConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    # Note: CausalConv3d expects (N, in_channels, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, depth, height, width)
    assert out_x.dtype == x.dtype
