import pytest
import torch

from wiskers.common.modules.gated_conv_1d import GatedConv1d  # Adjust the import based on your module structure


@pytest.mark.parametrize(
    "batch_size, length, in_channels",
    [
        (4, 32, 16),
        (4, 64, 8),
    ],
)
def test_gated_conv_1d(batch_size, length, in_channels):
    out_channels = 3
    kernel_size = 3
    net = GatedConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

    # Input tensor (N, C, L)
    x = torch.randn(batch_size, in_channels, length)
    out_x = net(x)

    # Assertions
    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (batch_size, out_channels, length)
    assert out_x.dtype == x.dtype
    assert torch.all(out_x >= 0)
