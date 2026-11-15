import torch

from wiskers.nn.losses import (
    L1Loss,
    L2Loss,
    MixedL1L2Loss,
    VQVAELoss,
    kl_divergence_standard_normal,
    ssim_with_loss,
)


def test_l1_loss() -> None:
    loss_fn = L1Loss()
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 4.0])
    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(1.5))


def test_l2_loss() -> None:
    loss_fn = L2Loss()
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 4.0])
    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(2.5))


def test_mixed_l1_l2_loss() -> None:
    loss_fn = MixedL1L2Loss(alpha=0.5)
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 4.0])
    loss = loss_fn(x, y)
    # 0.5 * 1.5 + 0.5 * 2.5 = 2.0
    assert torch.isclose(loss, torch.tensor(2.0))


def test_kl_divergence() -> None:
    mu = torch.zeros(2, 4)
    logvar = torch.zeros(2, 4)
    kl = kl_divergence_standard_normal(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0))


def test_ssim_with_loss() -> None:
    x = torch.rand(2, 3, 32, 32)
    val, loss = ssim_with_loss(x, x)
    assert torch.isclose(val, torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-3)


def test_vqvae_loss() -> None:
    loss_fn = VQVAELoss(reconstruction=MixedL1L2Loss(), ssim_weight=0.1)
    images = torch.rand(2, 3, 32, 32)
    recon = torch.rand(2, 3, 32, 32)
    vq_loss = torch.tensor(0.5)

    out = loss_fn(images, recon, vq_loss)
    assert "loss" in out
    assert "vq_loss" in out
    assert "reconstruction_loss" in out
    assert "ssim" in out
    assert "ssim_loss" in out
    assert out["loss"].ndim == 0

    # Test default reconstruction
    default_loss_fn = VQVAELoss()
    assert isinstance(default_loss_fn.reconstruction, MixedL1L2Loss)
    out_default = default_loss_fn(images, recon, vq_loss)
    assert out_default["loss"].ndim == 0
