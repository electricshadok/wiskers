"""
Tests for shared RL heads in wiskers.common.heads
(ContinuousActor, CategoricalActor, Critic, Reward, Done).
"""

import pytest
import torch

from wiskers.common.heads import (
    CategoricalActor,
    ContinuousActor,
    Critic,
    Done,
    Reward,
)

N = 4
T = 8
FEAT = 48
ACTION = 3
NUM_ACTIONS = 4
HIDDEN = 64
LAYERS = 2


@pytest.fixture
def actor():
    return ContinuousActor(feat_size=FEAT, action_size=ACTION, hidden_size=HIDDEN, num_layers=LAYERS)


@pytest.fixture
def feat():
    return torch.randn(N, FEAT)


# ──────────────────────────────────────────────────────────────────────────────
# ContinuousActor (Squashed Gaussian)
# ──────────────────────────────────────────────────────────────────────────────


def test_actor_forward_shapes(actor, feat):
    mu, std = actor(feat)
    assert mu.shape == (N, ACTION)
    assert std.shape == (N, ACTION)


def test_actor_mu_bounded(actor, feat):
    mu, _ = actor(feat)
    assert mu.min() > -1.0 and mu.max() < 1.0


def test_actor_std_positive(actor, feat):
    _, std = actor(feat)
    assert std.min() > 0.0


def test_actor_sample_shape(actor, feat):
    action, mu = actor.sample(feat)
    assert action.shape == (N, ACTION)
    assert mu.shape == (N, ACTION)


def test_actor_sample_bounded(actor, feat):
    action, _ = actor.sample(feat)
    assert action.min() > -1.0 and action.max() < 1.0


def test_actor_sample_stochastic(actor, feat):
    a1, _ = actor.sample(feat)
    a2, _ = actor.sample(feat)
    assert not torch.allclose(a1, a2)


def test_actor_gradients(actor, feat):
    action, _ = actor.sample(feat)
    loss = action.sum()
    loss.backward()
    for name, p in actor.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# CategoricalActor (Discrete Actions)
# ──────────────────────────────────────────────────────────────────────────────


def test_categorical_actor_forward_and_sample(feat):
    cat_actor = CategoricalActor(
        feat_size=FEAT,
        num_actions=NUM_ACTIONS,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
    )
    logits = cat_actor(feat)
    assert logits.shape == (N, NUM_ACTIONS)

    action, log_prob, entropy = cat_actor.sample(feat)
    assert action.shape == (N,)
    assert action.dtype == torch.int64
    assert ((action >= 0) & (action < NUM_ACTIONS)).all()
    assert log_prob.shape == (N,)
    assert entropy.shape == (N,)
    assert (entropy >= 0.0).all()

    loss = -(log_prob.sum() + entropy.sum())
    loss.backward()
    for name, p in cat_actor.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Critic & lambda_return
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def critic():
    return Critic(feat_size=FEAT, hidden_size=HIDDEN, num_layers=LAYERS)


def test_critic_forward_shape(critic, feat):
    value = critic(feat)
    assert value.shape == (N, 1)


def test_critic_output_unbounded(critic, feat):
    value = critic(feat)
    assert not torch.isnan(value).any()


def test_critic_gradients(critic, feat):
    value = critic(feat)
    value.sum().backward()
    for name, p in critic.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


def test_lambda_return_shape():
    rewards = torch.rand(T, N, 1)
    values = torch.rand(T + 1, N, 1)
    returns = Critic.lambda_return(rewards, values)
    assert returns.shape == (T, N, 1)


def test_lambda_return_mc_limit():
    rewards = torch.ones(T, 1, 1)
    values = torch.zeros(T + 1, 1, 1)
    returns = Critic.lambda_return(rewards, values, gamma=1.0, lam=1.0)
    assert abs(returns[T - 1].item() - 1.0) < 1e-4


def test_lambda_return_td_limit():
    rewards = torch.ones(T, 1, 1)
    values = torch.ones(T + 1, 1, 1) * 2.0
    returns = Critic.lambda_return(rewards, values, gamma=1.0, lam=0.0)
    assert torch.allclose(returns, torch.full_like(returns, 3.0), atol=1e-4)


def test_lambda_return_with_dones():
    rewards = torch.ones(4, 1, 1)
    values = torch.full((5, 1, 1), 10.0)
    dones = torch.zeros(4, 1, 1)
    dones[1] = 1.0  # Episode terminates at step t=1

    returns = Critic.lambda_return(rewards, values, gamma=0.99, lam=0.95, dones=dones)
    assert returns.shape == (4, 1, 1)
    # At t=1 where done=1, future bootstrap values must be masked out: G_1 = r_1 = 1.0
    assert torch.isclose(returns[1], torch.tensor(1.0))
    # And at t=0, G_0 only bootstraps from step 1 (V_1=10.0 and G_1=1.0), not steps 2..3
    expected_g0 = 1.0 + 0.99 * ((1.0 - 0.95) * 10.0 + 0.95 * 1.0)
    assert torch.isclose(returns[0], torch.tensor(expected_g0))


# ──────────────────────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def reward_head():
    return Reward(feat_size=FEAT, hidden_size=HIDDEN, num_layers=LAYERS)


def test_reward_forward_shape(reward_head, feat):
    r_hat = reward_head(feat)
    assert r_hat.shape == (N, 1)


def test_reward_output_unbounded(reward_head, feat):
    r_hat = reward_head(feat)
    assert not torch.isnan(r_hat).any()


def test_reward_loss_scalar(reward_head, feat):
    r_hat = reward_head(feat)
    r_target = torch.randn(N, 1)
    loss = Reward.loss(r_hat, r_target)
    assert loss.shape == ()


def test_reward_loss_zero_for_perfect_prediction():
    target = torch.randn(N, 1)
    loss = Reward.loss(target, target)
    assert abs(loss.item()) < 1e-6


def test_reward_gradients(reward_head, feat):
    r_hat = reward_head(feat)
    r_hat.sum().backward()
    for name, p in reward_head.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────────────


def test_done_head(feat):
    done_head = Done(feat_size=FEAT, hidden_size=HIDDEN, num_layers=LAYERS)
    logits = done_head(feat)
    probs = done_head.predict_proba(feat)

    assert logits.shape == (N, 1)
    assert probs.shape == (N, 1)
    assert (probs > 0.0).all() and (probs < 1.0).all()

    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    loss = Done.loss(logits, targets)
    assert loss.shape == ()
    loss.backward()
    for name, p in done_head.named_parameters():
        assert p.grad is not None, f"No gradient for: {name}"
