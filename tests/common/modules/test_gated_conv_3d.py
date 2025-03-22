import pytest
import torch

from wiskers.common.modules.gated_conv_3d import GatedConv3d


@pytest.mark.parametrize(
    "batch_size, depth, height, width, in_channels",
    [
        (2, 16, 32, 32, 8),
        (2, 8, 64, 64, 4),
    ],
)
def test_gated_conv_3d(batch_size, depth, height, width, in_channels):
    out_channels = 3
    kernel_size = 3
    net = GatedConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    # Input tensor (N, C, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    # Assertions
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, out_channels, depth, height, width)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)
