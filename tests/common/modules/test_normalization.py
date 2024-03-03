import pytest
import torch

from wiskers.common.modules.normalization import AdaIN


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 128, 128),
        (8, 1, 256, 256),
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
