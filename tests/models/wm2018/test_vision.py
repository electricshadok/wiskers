"""
Tests for Vision model (V) — World Models 2018 (Ha & Schmidhuber).
"""
import pytest
import torch

from wiskers.models.wm.wm2018.vision import Vision


# Smaller config for fast CPU tests
IMAGE_SIZE = (32, 32)
IN_CHANNELS = 3
LATENT_SIZE = 16
BLOCK_CHANNELS = [16, 32]
BLOCK_ATTENTIONS = [False, False]
BATCH = 4


@pytest.fixture
def vision():
    return Vision(
        image_size=IMAGE_SIZE,
        in_channels=IN_CHANNELS,
        latent_size=LATENT_SIZE,
        block_channels=BLOCK_CHANNELS,
        block_attentions=BLOCK_ATTENTIONS,
    )


@pytest.fixture
def frames():
    return torch.rand(BATCH, IN_CHANNELS, *IMAGE_SIZE)


# ------------------------------------------------------------------ #
#  encode                                                              #
# ------------------------------------------------------------------ #


def test_encode_output_shapes(vision, frames):
    z, mu, logvar = vision.encode(frames)
    assert z.shape == (BATCH, LATENT_SIZE)
    assert mu.shape == (BATCH, LATENT_SIZE)
    assert logvar.shape == (BATCH, LATENT_SIZE)


def test_encode_z_is_stochastic(vision, frames):
    """Two encode calls on the same input should give different z (reparameterization)."""
    z1, _, _ = vision.encode(frames)
    z2, _, _ = vision.encode(frames)
    assert not torch.allclose(z1, z2), "z should be stochastic — reparameterization adds noise"


def test_encode_mu_is_deterministic(vision, frames):
    """mu (the mean) must be deterministic for the same input."""
    _, mu1, _ = vision.encode(frames)
    _, mu2, _ = vision.encode(frames)
    assert torch.allclose(mu1, mu2), "mu should be deterministic"


# ------------------------------------------------------------------ #
#  encode_deterministic                                                #
# ------------------------------------------------------------------ #


def test_encode_deterministic_shape(vision, frames):
    mu = vision.encode_deterministic(frames)
    assert mu.shape == (BATCH, LATENT_SIZE)


def test_encode_deterministic_matches_mu(vision, frames):
    """encode_deterministic should return the same value as mu from encode."""
    _, mu, _ = vision.encode(frames)
    mu_det = vision.encode_deterministic(frames)
    assert torch.allclose(mu, mu_det), "encode_deterministic must match encode's mu"


def test_encode_deterministic_no_grad(vision, frames):
    """encode_deterministic must not accumulate gradients (frozen Stage 2)."""
    frames_grad = frames.requires_grad_(True)
    mu = vision.encode_deterministic(frames_grad)
    assert mu.grad_fn is None, "encode_deterministic should produce a no-grad tensor"


# ------------------------------------------------------------------ #
#  decode                                                              #
# ------------------------------------------------------------------ #


def test_decode_output_shape(vision):
    z = torch.randn(BATCH, LATENT_SIZE)
    x_hat = vision.decode(z)
    assert x_hat.shape == (BATCH, IN_CHANNELS, *IMAGE_SIZE)


def test_decode_output_range(vision):
    """Decoder applies sigmoid — output must be in [0, 1]."""
    z = torch.randn(BATCH, LATENT_SIZE)
    x_hat = vision.decode(z)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "decode output must be in [0, 1]"


# ------------------------------------------------------------------ #
#  forward                                                             #
# ------------------------------------------------------------------ #


def test_forward_output_shapes(vision, frames):
    x_hat, mu, logvar = vision(frames)
    assert x_hat.shape == (BATCH, IN_CHANNELS, *IMAGE_SIZE)
    assert mu.shape == (BATCH, LATENT_SIZE)
    assert logvar.shape == (BATCH, LATENT_SIZE)


def test_forward_output_range(vision, frames):
    x_hat, _, _ = vision(frames)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


def test_forward_gradients_flow(vision, frames):
    """Loss must produce valid gradients through the full VAE."""
    import torch.nn.functional as F

    frames = frames.clone().requires_grad_(False)
    x_hat, mu, logvar = vision(frames)

    recon_loss = F.mse_loss(x_hat, frames)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    loss.backward()

    # All encoder/decoder params should have gradients
    for name, p in vision.named_parameters():
        assert p.grad is not None, f"No gradient for param: {name}"


# ------------------------------------------------------------------ #
#  paper defaults smoke test                                           #
# ------------------------------------------------------------------ #


def test_paper_defaults_forward():
    """Verify the paper-default config (64x64, z=32) runs without error."""
    model = Vision()  # paper defaults
    x = torch.rand(2, 3, 64, 64)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == (2, 3, 64, 64)
    assert mu.shape == (2, 32)
