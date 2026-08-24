import torch

from wiskers.models.world_model.mdn_rnn import MDNRNN, MDNHead


# ---------------------------------------------------------------------------
# MDNHead
# ---------------------------------------------------------------------------


def test_mdn_head_pi_sums_to_one():
    """Mixture weights must sum to 1 across the Gaussian dimension."""
    head = MDNHead(hidden_size=64, latent_size=8, num_gaussians=5)
    h = torch.randn(2, 5, 64)
    pi, _, _ = head(h)

    sums = pi.sum(dim=-1)  # [N, T]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "pi does not sum to 1"


def test_mdn_head_sigma_positive():
    """Standard deviations must always be strictly positive (clamp ≥ 1e-6)."""
    head = MDNHead(hidden_size=64, latent_size=8, num_gaussians=5)
    h = torch.randn(2, 5, 64)
    _, _, sigma = head(h)

    assert (sigma > 0).all(), "sigma contains non-positive values"


# ---------------------------------------------------------------------------
# MDNRNN
# ---------------------------------------------------------------------------


def test_mdnrnn_output_shapes():
    """MDNRNN forward pass should return tensors of the correct shapes, including done."""
    batch_size, seq_len, latent_size, action_size, hidden_size, num_gaussians, num_layers = (
        4, 10, 32, 3, 256, 5, 1
    )

    model = MDNRNN(
        latent_size=latent_size,
        action_size=action_size,
        hidden_size=hidden_size,
        num_gaussians=num_gaussians,
        num_layers=num_layers,
    )

    z = torch.randn(batch_size, seq_len, latent_size)
    a = torch.randn(batch_size, seq_len, action_size)

    pi, mu, sigma, done, (h_n, c_n) = model(z, a)

    assert pi.shape == (batch_size, seq_len, num_gaussians)
    assert mu.shape == (batch_size, seq_len, num_gaussians, latent_size)
    assert sigma.shape == (batch_size, seq_len, num_gaussians, latent_size)
    assert done.shape == (batch_size, seq_len, 1)
    assert (done >= 0).all() and (done <= 1).all(), "done values outside [0, 1]"
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)


def test_mdnrnn_stateful_rollout():
    """Passing hidden state between calls should work (auto-regressive rollout)."""
    model = MDNRNN(latent_size=16, action_size=2, hidden_size=64, num_gaussians=3)
    device = torch.device("cpu")

    batch_size = 2
    hidden = model.get_initial_hidden(batch_size, device)

    for _ in range(5):
        z = torch.randn(batch_size, 1, 16)
        a = torch.randn(batch_size, 1, 2)
        pi, mu, sigma, done, hidden = model(z, a, hidden)

    # Hidden state should still have valid shapes after 5 steps
    h_n, c_n = hidden
    assert h_n.shape == (1, batch_size, 64)
    assert c_n.shape == (1, batch_size, 64)


def test_mdnrnn_get_initial_hidden():
    """get_initial_hidden should return zero tensors of the right shape."""
    model = MDNRNN(latent_size=32, action_size=4, hidden_size=128, num_layers=2)
    h_0, c_0 = model.get_initial_hidden(batch_size=3, device=torch.device("cpu"))

    assert h_0.shape == (2, 3, 128)
    assert c_0.shape == (2, 3, 128)
    assert torch.all(h_0 == 0)
    assert torch.all(c_0 == 0)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_losses():
    """mdn_loss and done_loss should be positive scalars with gradients flowing to all parameters."""
    model = MDNRNN(latent_size=16, action_size=2, hidden_size=64, num_gaussians=5)
    z = torch.randn(4, 10, 16)
    a = torch.randn(4, 10, 2)

    pi, mu, sigma, done, _ = model(z, a)

    z_next = torch.randn(4, 10, 16)
    mdn_loss = MDNRNN.mdn_loss(pi, mu, sigma, z_next)
    done_target = torch.randint(0, 2, (4, 10)).float()
    done_loss = MDNRNN.done_loss(done, done_target)
    loss = mdn_loss + done_loss

    assert mdn_loss.shape == () and mdn_loss.item() > 0, "mdn_loss should be a positive scalar"
    assert done_loss.shape == () and done_loss.item() > 0, "done_loss should be a positive scalar"

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
