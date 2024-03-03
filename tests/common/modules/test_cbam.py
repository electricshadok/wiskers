import pytest
import torch

from wiskers.common.modules.cbam import CBAM, ChannelAttention, SpatialAttention


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 8, 64, 64),
        (8, 8, 32, 32),
    ],
)
def test_channel_attention(batch_size, in_channels, height, width):
    net = ChannelAttention(in_channels, 4)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, in_channels, 1, 1)
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 8, 64, 64),
        (8, 8, 32, 32),
    ],
)
def test_spatial_attention(batch_size, in_channels, height, width):
    net = SpatialAttention()
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, 1, height, width)
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 8, 64, 64),
        (8, 8, 32, 32),
    ],
)
def test_cbam(batch_size, in_channels, height, width):
    net = CBAM(in_channels, 4)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, in_channels, height, width)
    assert out_x.dtype == x.dtype
