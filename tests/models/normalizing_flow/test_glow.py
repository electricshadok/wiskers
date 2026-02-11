import pytest
import torch

from wiskers.models.normalizing_flow.glow import ActNorm, InvertibleConv1x1


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 32, 32),
        (8, 1, 32, 32),
    ],
)
def test_invertible_conv1x1(batch_size, in_channels, height, width):
    net = InvertibleConv1x1(num_channels=in_channels)
    x = torch.randn(batch_size, in_channels, height, width)
    z, logdet = net(x, logdet=None, reverse=False)

    assert z.shape == (batch_size, in_channels, height, width)
    assert z.dtype == x.dtype

    # w should be an orthogonal matrix -> det(W) = 1 or -1
    det_w = torch.linalg.det(net.w).item()
    assert abs(det_w) == pytest.approx(1.0, rel=1e-6)

    # test function is invertible
    x_reconstructed, _ = net(z, logdet=None, reverse=True)
    torch.testing.assert_close(x, x_reconstructed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 3, 32, 32),
        (8, 1, 32, 32),
    ],
)
def test_actnorm(batch_size, in_channels, height, width):
    net = ActNorm(num_channels=in_channels)
    x = torch.randn(batch_size, in_channels, height, width)

    # forward (this will initialize parameters on the first batch)
    y, logdet = net(x, logdet=None, reverse=False)

    assert y.shape == (batch_size, in_channels, height, width)
    assert y.dtype == x.dtype

    # After initialization and forward, each channel should have ~0 mean and ~1 std
    mean_per_channel = y.mean(dim=(0, 2, 3))
    std_per_channel = y.std(dim=(0, 2, 3), unbiased=False)

    torch.testing.assert_close(
        mean_per_channel, torch.zeros_like(mean_per_channel), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        std_per_channel, torch.ones_like(std_per_channel), atol=1e-5, rtol=1e-5
    )

    # test invertibility
    x_reconstructed, _ = net(y, logdet=None, reverse=True)
    torch.testing.assert_close(x, x_reconstructed, atol=1e-5, rtol=1e-5)
