import pytest
import torch

from wiskers.common.modules.gated_conv_3d import GatedConv3d, GatedConvTranspose3d


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
    net = GatedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=1,
    )

    # Input tensor (N, C, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, depth, height, width)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)


@pytest.mark.parametrize(
    "batch_size, depth, height, width, in_channels",
    [
        (2, 8, 8, 8, 4),
        (1, 4, 16, 16, 8),
    ],
)
def test_gated_conv_transpose_3d(batch_size, depth, height, width, in_channels):
    out_channels = 6
    kernel_size = 4
    stride = 2
    padding = 1
    output_padding = 0

    net = GatedConvTranspose3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    # Input tensor (N, C, D, H, W)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    # Expect spatial dims to double: D, H, W
    assert out_x.shape == (
        batch_size,
        out_channels,
        depth * 2,
        height * 2,
        width * 2,
    )
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)
