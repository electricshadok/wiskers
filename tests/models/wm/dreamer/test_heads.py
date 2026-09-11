"""
Tests for Dreamer heads (Actor, Critic, Reward).

Kept simple: shape checks + key behavioural invariants.
No trainer, no Lightning — pure nn.Module tests.
"""
import pytest
import torch

from wiskers.models.wm.dreamer.heads import Actor, Critic, Reward


# ── Shared config (small, fast on CPU) ────────────────────────────────────────

N = 4            # batch size
T = 8            # imagined sequence length
FEAT = 48        # hidden_size + stoch_size (e.g. 32 + 16)
ACTION = 3       # action dims
HIDDEN = 64      # MLP hidden size
LAYERS = 2       # number of hidden layers


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def actor():
    return Actor(feat_size=FEAT, action_size=ACTION, hidden_size=HIDDEN, num_layers=LAYERS)


@pytest.fixture
def feat():
    return torch.randn(N, FEAT)


def test_actor_forward_shapes(actor, feat):
    """forward() should return (mu, std) both of shape [N, action_size]."""
    mu, std = actor(feat)
    assert mu.shape == (N, ACTION)
    assert std.shape == (N, ACTION)


def test_actor_mu_bounded(actor, feat):
    """mu is squashed with tanh — must be strictly in (-1, 1)."""
    mu, _ = actor(feat)
    assert mu.min() > -1.0 and mu.max() < 1.0


def test_actor_std_positive(actor, feat):
    """std must always be strictly positive."""
    _, std = actor(feat)
    assert std.min() > 0.0


def test_actor_sample_shape(actor, feat):
    """sample() should return (action, mu) both of shape [N, action_size]."""
    action, mu = actor.sample(feat)
    assert action.shape == (N, ACTION)
    assert mu.shape == (N, ACTION)


def test_actor_sample_bounded(actor, feat):
    """Sampled actions are tanh-squashed — must be in (-1, 1)."""
    action, _ = actor.sample(feat)
    assert action.min() > -1.0 and action.max() < 1.0


def test_actor_sample_stochastic(actor, feat):
    """Two sample() calls from the same feat should give different actions."""
    a1, _ = actor.sample(feat)
    a2, _ = actor.sample(feat)
    assert not torch.allclose(a1, a2)


def test_actor_gradients(actor, feat):
    """Gradients must flow back through sample() into actor weights."""
    action, _ = actor.sample(feat)
    loss = action.sum()
    loss.backward()
    for name, p in actor.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def critic():
    return Critic(feat_size=FEAT, hidden_size=HIDDEN, num_layers=LAYERS)


def test_critic_forward_shape(critic, feat):
    """forward() should return a scalar value per sample [N, 1]."""
    value = critic(feat)
    assert value.shape == (N, 1)


def test_critic_output_unbounded(critic, feat):
    """Value estimates are not squashed — output can be any real number."""
    value = critic(feat)
    # Just confirm it's a real tensor with no NaNs
    assert not torch.isnan(value).any()


def test_critic_gradients(critic, feat):
    """Gradients must flow back through critic into its weights."""
    value = critic(feat)
    value.sum().backward()
    for name, p in critic.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


def test_lambda_return_shape():
    """lambda_return should produce [T, N, 1] returns."""
    rewards = torch.rand(T, N, 1)
    values = torch.rand(T + 1, N, 1)  # T steps + bootstrap
    returns = Critic.lambda_return(rewards, values)
    assert returns.shape == (T, N, 1)


def test_lambda_return_mc_limit():
    """
    With lam=1 and gamma=1, lambda-return collapses to undiscounted Monte Carlo.
    G_t = sum of all future rewards → last step return = r_{T-1}.
    """
    rewards = torch.ones(T, 1, 1)
    values = torch.zeros(T + 1, 1, 1)
    returns = Critic.lambda_return(rewards, values, gamma=1.0, lam=1.0)
    # G_0 should equal T (sum of T unit rewards), G_{T-1} should equal 1
    assert abs(returns[T - 1].item() - 1.0) < 1e-4


def test_lambda_return_td_limit():
    """
    With lam=0 and gamma=1, lambda-return collapses to 1-step TD.
    G_t = r_t + V(s_{t+1}).
    """
    rewards = torch.ones(T, 1, 1)
    values = torch.ones(T + 1, 1, 1) * 2.0
    returns = Critic.lambda_return(rewards, values, gamma=1.0, lam=0.0)
    # Each return should be r_t + V(s_{t+1}) = 1 + 2 = 3
    assert torch.allclose(returns, torch.full_like(returns, 3.0), atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def reward_head():
    return Reward(feat_size=FEAT, hidden_size=HIDDEN, num_layers=LAYERS)


def test_reward_forward_shape(reward_head, feat):
    """forward() should return a scalar reward per sample [N, 1]."""
    r_hat = reward_head(feat)
    assert r_hat.shape == (N, 1)


def test_reward_output_unbounded(reward_head, feat):
    """Reward predictions are not squashed — can be any real number."""
    r_hat = reward_head(feat)
    assert not torch.isnan(r_hat).any()


def test_reward_loss_scalar(reward_head, feat):
    """loss() should return a scalar."""
    r_hat = reward_head(feat)
    r_target = torch.randn(N, 1)
    loss = Reward.loss(r_hat, r_target)
    assert loss.shape == ()


def test_reward_loss_zero_for_perfect_prediction(reward_head):
    """MSE loss should be 0 when prediction exactly matches target."""
    target = torch.randn(N, 1)
    loss = Reward.loss(target, target)
    assert abs(loss.item()) < 1e-6


def test_reward_gradients(reward_head, feat):
    """Gradients must flow back through forward() into reward head weights."""
    r_hat = reward_head(feat)
    r_hat.sum().backward()
    for name, p in reward_head.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"
