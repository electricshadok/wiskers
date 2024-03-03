import pytest
import torch

from wiskers.diffusion.modules.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 16, 32, 32, 128),
        (8, 1, 32, 64, 64, 128),
    ],
)
def test_down_block_2d(batch_size, in_channels, out_channels, height, width, time_dim):
    net = DownBlock2D(in_channels, out_channels, time_dim)
    x = torch.randn(batch_size, in_channels, height, width)
    te = torch.randn(batch_size, time_dim)
    out_x = net(x, te)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] // 2,
        x.shape[3] // 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 16, 32, 32, 128),
        (8, 1, 32, 64, 64, 128),
    ],
)
def test_attn_down_block_2d(batch_size, in_channels, out_channels, height, width, time_dim):
    net = AttnDownBlock2D(in_channels, out_channels, time_dim)
    x = torch.randn(batch_size, in_channels, height, width)
    te = torch.randn(batch_size, time_dim)
    out_x = net(x, te)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] // 2,
        x.shape[3] // 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 16, 32, 32, 128),
        (8, 1, 32, 64, 64, 128),
    ],
)
def test_up_block_2d(batch_size, in_channels, out_channels, height, width, time_dim):
    net = UpBlock2D(in_channels, in_channels, out_channels, time_dim)
    x = torch.randn(batch_size, in_channels, height, width)
    te = torch.randn(batch_size, time_dim)
    skip_x = torch.randn(batch_size, in_channels, height * 2, width * 2)
    out_x = net(x, skip_x, te)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] * 2,
        x.shape[3] * 2,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 16, 32, 32, 128),
        (8, 1, 32, 64, 64, 128),
    ],
)
def test_attn_up_block_2d(batch_size, in_channels, out_channels, height, width, time_dim):
    net = AttnUpBlock2D(in_channels, in_channels, out_channels, time_dim)
    x = torch.randn(batch_size, in_channels, height, width)
    te = torch.randn(batch_size, time_dim)
    skip_x = torch.randn(batch_size, in_channels, height * 2, width * 2)
    out_x = net(x, skip_x, te)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (
        batch_size,
        out_channels,
        x.shape[2] * 2,
        x.shape[3] * 2,
    )
    assert out_x.dtype == x.dtype
