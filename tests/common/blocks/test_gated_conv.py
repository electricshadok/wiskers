import pytest
import torch

from wiskers.common.blocks.gated_conv import (
    GatedConv1d,
    GatedConv2d,
    GatedConv3d,
    GatedConvTranspose1d,
    GatedConvTranspose2d,
    GatedConvTranspose3d,
)


# ---------------------------------------------------------------------------
# 1D Gated Convolution Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, length, in_channels",
    [
        (4, 32, 16),
        (4, 64, 8),
    ],
)
def test_gated_conv_1d(batch_size, length, in_channels):
    out_channels = 3
    net = GatedConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )
    x = torch.randn(batch_size, in_channels, length)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, length)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)


@pytest.mark.parametrize(
    "batch_size, length, in_channels",
    [
        (4, 16, 8),
        (2, 8, 4),
    ],
)
def test_gated_conv_transpose_1d(batch_size, length, in_channels):
    out_channels = 3
    net = GatedConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
    )
    x = torch.randn(batch_size, in_channels, length)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, length * 2)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)


# ---------------------------------------------------------------------------
# 2D Gated Convolution Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, height, width, in_channels",
    [
        (4, 32, 32, 16),
        (4, 64, 64, 8),
    ],
)
def test_gated_conv_2d(batch_size, height, width, in_channels):
    out_channels = 3
    net = GatedConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )
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
    net = GatedConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (batch_size, out_channels, height * 2, width * 2)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)


# ---------------------------------------------------------------------------
# 3D Gated Convolution Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, depth, height, width, in_channels",
    [
        (2, 16, 32, 32, 8),
        (2, 8, 64, 64, 4),
    ],
)
def test_gated_conv_3d(batch_size, depth, height, width, in_channels):
    out_channels = 3
    net = GatedConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )
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
    net = GatedConvTranspose3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        output_padding=0,
    )
    x = torch.randn(batch_size, in_channels, depth, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        depth * 2,
        height * 2,
        width * 2,
    )
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)

