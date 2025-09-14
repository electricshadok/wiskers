import pytest
import torch

from wiskers.common.blocks.gated_conv_2d import GatedConv2d, GatedConvTranspose2d


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

    assert out_x.shape == (batch_size, out_channels, height, width)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)


@pytest.mark.parametrize(
    "batch_size, height, width, in_channels",
    [
        (4, 16, 16, 8),
        (2, 8, 8, 4),
    ],
)
def test_gated_conv_transpose_2d(batch_size, height, width, in_channels):
    out_channels = 3
    kernel_size = 4
    stride = 2
    padding = 1
    output_padding = 0

    net = GatedConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    # Input tensor (N, C, H, W)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    # Expect output spatial dimensions doubled (same as in 1D test)
    assert out_x.shape == (batch_size, out_channels, height * 2, width * 2)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)
