import torch

from wiskers.models.world_model.controller import Controller


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


def test_controller_output_shape():
    """Controller should return an action of shape [N, action_size]."""
    latent_size, hidden_size, action_size, batch_size = 32, 256, 3, 4

    model = Controller(latent_size=latent_size, hidden_size=hidden_size, action_size=action_size)
    z = torch.randn(batch_size, latent_size)
    h = torch.randn(batch_size, hidden_size)

    a = model(z, h)

    assert a.shape == (batch_size, action_size), f"Unexpected action shape: {a.shape}"


def test_controller_action_range():
    """Actions must lie in [-1, 1] — enforced by tanh output."""
    model = Controller(latent_size=32, hidden_size=256, action_size=3)
    z = torch.randn(16, 32) * 100  # extreme inputs to stress-test tanh
    h = torch.randn(16, 256) * 100

    a = model(z, h)

    assert (a >= -1).all() and (a <= 1).all(), "Actions outside [-1, 1]"


def test_controller_backward():
    """Gradients must flow to all parameters."""
    model = Controller(latent_size=32, hidden_size=64, action_size=2)
    z = torch.randn(4, 32)
    h = torch.randn(4, 64)

    a = model(z, h)
    loss = a.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# Param vector helpers (used by CMA-ES style optimisers)
# ---------------------------------------------------------------------------


def test_param_vector_roundtrip():
    """get_param_vector → set_param_vector should recover identical parameters."""
    model = Controller(latent_size=32, hidden_size=64, action_size=3)

    original = model.get_param_vector().clone()
    # Corrupt the model weights, then restore from the vector
    for p in model.parameters():
        p.data.zero_()

    model.set_param_vector(original)
    restored = model.get_param_vector()

    assert torch.allclose(original, restored), "Parameters not restored correctly"


def test_num_params_matches_vector_length():
    """num_params() should equal the length of get_param_vector()."""
    model = Controller(latent_size=16, hidden_size=32, action_size=4)

    assert model.num_params() == model.get_param_vector().numel()
