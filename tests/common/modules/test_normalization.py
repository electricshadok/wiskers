import pytest
import torch

from wiskers.common.modules.normalization import AdaIN, DyT


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 64, 64),
        (8, 1, 32, 32),
    ],
)
def test_AdaIN(batch_size, in_channels, height, width):
    net = AdaIN()
    x = torch.randn(batch_size, in_channels, height, width)
    y = torch.randn(batch_size, in_channels, height, width)

    out_x = net(x, y)
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 64, 64),
        (8, 1, 32, 32),
    ],
)
def test_DyT_element_wise(batch_size, in_channels, height, width):
    net = DyT(normalized_shape=(in_channels, height, width), channels_last=True)

    x = torch.randn(batch_size, in_channels, height, width)

    out_x = net(x)

    # Assertions
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 64, 64),
        (8, 1, 32, 32),
    ],
)
def test_DyT_channel_wise(batch_size, in_channels, height, width):
    net = DyT(normalized_shape=in_channels, channels_last=False)

    x = torch.randn(batch_size, in_channels, height, width)

    out_x = net(x)

    # Assertions
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype
