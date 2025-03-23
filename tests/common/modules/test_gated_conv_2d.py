import pytest
import torch

from wiskers.common.modules.gated_conv_2d import GatedConv2d


@pytest.mark.parametrize(
    "batch_size, height, width, in_channels",
    [
        (4, 32, 32, 16),
        (4, 64, 64, 8),
    ],
)
def test_gated_conv_2d(batch_size, height, width, in_channels):
    out_channels = 3
    kernel_size = 3
    net = GatedConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=1,
    )

    # Input tensor (N, C, H, W)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    # Assertions
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, out_channels, height, width)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)
