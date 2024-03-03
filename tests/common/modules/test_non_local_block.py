import pytest
import torch

from wiskers.common.modules.non_local_block import NonLocalBlock


@pytest.mark.parametrize(
    "batch_size, in_channels, length, height, width",
    [
        (8, 32, 4, 14, 14),
    ],
)
def test_non_local_block(batch_size, in_channels, length, height, width):
    x = torch.randn(batch_size, in_channels, length, height, width)
    net = NonLocalBlock(in_channels, "embedded_gaussian")
    out_x = net(x)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == x.shape
    assert out_x.dtype == x.dtype
