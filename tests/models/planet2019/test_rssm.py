"""
Tests for RSSM — PlaNet (Hafner et al., 2019).

Kept intentionally simple: shape checks + basic behavioural invariants.
No trainer, no Lightning — pure nn.Module tests.
"""
import pytest
import torch

from wiskers.models.wm.planet2019.rssm import RSSM, RSSMState


# ── Shared config (small, fast on CPU) ────────────────────────────────────────

N = 4          # batch size
T = 8          # sequence length
STOCH = 16     # stochastic state size
HIDDEN = 64    # deterministic state size
ACTION = 3     # action dims
EMBED = 128    # obs embedding dims (from CNN encoder)


@pytest.fixture
def rssm():
    return RSSM(
        stoch_size=STOCH,
        hidden_size=HIDDEN,
        action_size=ACTION,
        embed_size=EMBED,
    )


@pytest.fixture
def init_state():
    return RSSMState.zeros(N, HIDDEN, STOCH, device=torch.device("cpu"))


@pytest.fixture
def actions():
    return torch.randn(N, T, ACTION)


@pytest.fixture
def obs_embeds():
    return torch.randn(N, T, EMBED)


# ── RSSMState ─────────────────────────────────────────────────────────────────


def test_rssm_state_combined_shape(init_state):
    """combined should concat h and s along the last dim."""
    assert init_state.combined.shape == (N, HIDDEN + STOCH)


def test_rssm_state_zeros(init_state):
    assert init_state.h.shape == (N, HIDDEN)
    assert init_state.s.shape == (N, STOCH)
    assert init_state.h.sum() == 0.0
    assert init_state.s.sum() == 0.0


# ── prior_step ────────────────────────────────────────────────────────────────


def test_prior_step_shapes(rssm, init_state):
    """prior_step should return a valid state and Gaussian params."""
    a = torch.randn(N, ACTION)
    state, mu_p, logvar_p = rssm.prior_step(init_state, a)

    assert state.h.shape == (N, HIDDEN)
    assert state.s.shape == (N, STOCH)
    assert mu_p.shape == (N, STOCH)
    assert logvar_p.shape == (N, STOCH)


def test_prior_step_stochastic(rssm, init_state):
    """Two prior steps from the same state should give different s (reparameterization)."""
    a = torch.randn(N, ACTION)
    s1, _, _ = rssm.prior_step(init_state, a)
    s2, _, _ = rssm.prior_step(init_state, a)
    assert not torch.allclose(s1.s, s2.s)


# ── posterior_step ────────────────────────────────────────────────────────────


def test_posterior_step_shapes(rssm, init_state):
    """posterior_step should return state + both prior and posterior params."""
    a = torch.randn(N, ACTION)
    e = torch.randn(N, EMBED)
    state, mu_p, logvar_p, mu_q, logvar_q = rssm.posterior_step(init_state, a, e)

    assert state.h.shape == (N, HIDDEN)
    assert state.s.shape == (N, STOCH)
    assert mu_p.shape == (N, STOCH)
    assert mu_q.shape == (N, STOCH)
    assert logvar_p.shape == (N, STOCH)
    assert logvar_q.shape == (N, STOCH)


def test_posterior_differs_from_prior(rssm, init_state):
    """Posterior mu should differ from prior mu (observation refines the estimate)."""
    a = torch.randn(N, ACTION)
    e = torch.randn(N, EMBED)
    _, mu_p, _, mu_q, _ = rssm.posterior_step(init_state, a, e)
    assert not torch.allclose(mu_p, mu_q)


# ── observe_rollout ───────────────────────────────────────────────────────────


def test_observe_rollout_shapes(rssm, actions, obs_embeds):
    """observe_rollout should return T states and stacked prior/posterior tensors."""
    states, mu_p, logvar_p, mu_q, logvar_q = rssm.observe_rollout(obs_embeds, actions)

    assert len(states) == T
    assert mu_p.shape == (N, T, STOCH)
    assert logvar_p.shape == (N, T, STOCH)
    assert mu_q.shape == (N, T, STOCH)
    assert logvar_q.shape == (N, T, STOCH)


def test_observe_rollout_state_shapes(rssm, actions, obs_embeds):
    states, *_ = rssm.observe_rollout(obs_embeds, actions)
    for state in states:
        assert state.h.shape == (N, HIDDEN)
        assert state.s.shape == (N, STOCH)


# ── imagine_rollout ───────────────────────────────────────────────────────────


def test_imagine_rollout_shapes(rssm, init_state, actions):
    """imagine_rollout should return T states using only the prior."""
    states, mu_p, logvar_p = rssm.imagine_rollout(init_state, actions)

    assert len(states) == T
    assert mu_p.shape == (N, T, STOCH)
    assert logvar_p.shape == (N, T, STOCH)


# ── kl_loss ───────────────────────────────────────────────────────────────────


def test_kl_loss_scalar(rssm, actions, obs_embeds):
    """kl_loss should return a scalar."""
    _, mu_p, logvar_p, mu_q, logvar_q = rssm.observe_rollout(obs_embeds, actions)
    loss = RSSM.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
    assert loss.shape == ()


def test_kl_loss_positive(rssm, actions, obs_embeds):
    """KL divergence is always >= 0, and free_nats floor keeps it >= free_nats."""
    _, mu_p, logvar_p, mu_q, logvar_q = rssm.observe_rollout(obs_embeds, actions)
    free_nats = 3.0
    loss = RSSM.kl_loss(mu_q, logvar_q, mu_p, logvar_p, free_nats=free_nats)
    assert loss.item() >= free_nats


def test_kl_loss_identical_distributions():
    """KL(q||p) = 0 when q == p; with free_nats the loss equals free_nats."""
    mu = torch.zeros(2, 5, 16)
    logvar = torch.zeros(2, 5, 16)
    free_nats = 3.0
    loss = RSSM.kl_loss(mu, logvar, mu, logvar, free_nats=free_nats)
    assert abs(loss.item() - free_nats) < 1e-4


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_flow_through_observe_rollout(rssm, actions, obs_embeds):
    """Backward through observe_rollout must produce gradients for all RSSM params."""
    states, mu_p, logvar_p, mu_q, logvar_q = rssm.observe_rollout(obs_embeds, actions)
    loss = RSSM.kl_loss(mu_q, logvar_q, mu_p, logvar_p)
    loss.backward()

    for name, p in rssm.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"
