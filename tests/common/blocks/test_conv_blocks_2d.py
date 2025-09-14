import pytest
import torch

from wiskers.common.blocks.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DoubleConv2D,
    DownBlock2D,
    ResDoubleConv2D,
    SeparableConv2d,
    UpBlock2D,
)


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 16, 32, 32),
        (8, 1, 32, 64, 64),
    ],
)
def test_attn_down_block_2d(batch_size, in_channels, out_channels, height, width):
    net = AttnDownBlock2D(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] // 2,
        x.shape[3] // 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 16, 32, 32),
        (8, 1, 32, 64, 64),
    ],
)
def test_attn_up_block_2d(batch_size, in_channels, out_channels, height, width):
    net = AttnUpBlock2D(in_channels, in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    skip_x = torch.randn(batch_size, in_channels, height * 2, width * 2)
    out_x = net(x, skip_x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] * 2,
        x.shape[3] * 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [(4, 3, 16, 32, 32), (8, 1, 32, 64, 64)],
)
def test_double_conv_2d(batch_size, in_channels, out_channels, height, width):
    net = DoubleConv2D(in_channels=in_channels, out_channels=out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2],
        x.shape[3],
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 16, 32, 32),
        (8, 1, 32, 64, 64),
    ],
)
def test_down_block_2d(batch_size, in_channels, out_channels, height, width):
    net = DownBlock2D(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] // 2,
        x.shape[3] // 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, channels, height, width",
    [(4, 3, 32, 32), (8, 1, 64, 64)],
)
def test_residual_double_conv_2d(batch_size, channels, height, width):
    net = ResDoubleConv2D(channels)
    x = torch.randn(batch_size, channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        channels,
        x.shape[2],
        x.shape[3],
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [(4, 3, 16, 32, 32), (8, 1, 32, 64, 64)],
)
def test_separable_conv2d(batch_size, in_channels, out_channels, height, width):
    net = SeparableConv2d(in_channels, out_channels, kernel_size=3)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] - 2,  # -2 because using kernel_size=3
        x.shape[3] - 2,  # -2 because using kernel_size=3
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 16, 32, 32),
        (8, 1, 32, 64, 64),
    ],
)
def test_up_block_2d(batch_size, in_channels, out_channels, height, width):
    net = UpBlock2D(in_channels, in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    skip_x = torch.randn(batch_size, in_channels, height * 2, width * 2)
    out_x = net(x, skip_x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] * 2,
        x.shape[3] * 2,
    )
    assert out_x.dtype == x.dtype
