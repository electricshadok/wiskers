import pytest
import torch

from wiskers.common.modules.se_block_2d import SEBlock


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 16, 64, 64),
        (8, 16, 32, 32),
    ],
)
def test_se_block(batch_size, in_channels, height, width):
    net = SEBlock(in_channels, squeeze_channels=max(1, in_channels // 2))
    x = torch.randn(batch_size, in_channels, height, width)

    out_x = net(x)

    assert out_x.shape == x.shape
    assert out_x.dtype == torch.float32
