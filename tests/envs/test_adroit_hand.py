import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs import TorchGymEnv
from wiskers.envs.adroit_hand import (
    AVAILABLE_TASKS,
    AdroitHandHeuristicPolicy,
    AdroitHandNeuralPolicy,
    AdroitHandPolicy,
    AdroitHandRandomPolicy,
    make_adroit_hand_env,
)


@pytest.fixture(autouse=True)
def mock_mujoco_renderer(monkeypatch):
    """Bypass OpenGL/GLFW window creation during unit tests while keeping real MuJoCo physics."""
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )


@pytest.fixture
def adroit_env():
    env = make_adroit_hand_env(task="door", image_size=(64, 64))
    yield env
    env.close()


def test_make_adroit_hand_env_observation_shape(adroit_env):
    obs, info = adroit_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8

    action = adroit_env.action_space.sample()
    assert action.shape == (28,)
    next_obs, reward, terminated, truncated, info = adroit_env.step(action)
    assert next_obs.shape == (64, 64, 3)
    assert next_obs.dtype == np.uint8
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_make_adroit_hand_env_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        make_adroit_hand_env(task="non_existent_task")


def test_adroit_hand_available_tasks():
    assert "door" in AVAILABLE_TASKS
    assert "hammer" in AVAILABLE_TASKS
    assert "pen" in AVAILABLE_TASKS
    assert "relocate" in AVAILABLE_TASKS


def test_adroit_random_policy(adroit_env):
    torch_env = TorchGymEnv(adroit_env)
    policy = AdroitHandRandomPolicy(env=torch_env)
    obs, _ = torch_env.reset()
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (28,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_adroit_heuristic_policy(adroit_env):
    torch_env = TorchGymEnv(adroit_env)
    policy = AdroitHandHeuristicPolicy(momentum=0.8, noise_scale=0.2, env=torch_env)
    obs, _ = torch_env.reset()
    act1 = policy(obs, step=0)
    act2 = policy(obs, step=1)
    assert act1.shape == (28,)
    assert act2.shape == (28,)
    assert torch.all(act1 >= -1.0) and torch.all(act1 <= 1.0)
    assert torch.all(act2 >= -1.0) and torch.all(act2 <= 1.0)

    policy.reset()
    assert torch.all(policy.current_action == 0.0)


def test_adroit_hand_collector_integration(adroit_env):
    policy = AdroitHandHeuristicPolicy()
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=TorchGymEnv(adroit_env),
        policy=policy,
        transform=transform,
        num_episodes=1,
        min_steps=5,
        max_steps=10,
        verbose=False,
    )
    buffer = ReplayBuffer(capacity_episodes=5)
    collected_steps = collector.collect(buffer)

    assert collected_steps == 10
    assert buffer.num_episodes == 1
    assert buffer.num_steps == 10
    assert buffer[0].obs.shape == (11, 64, 64, 3)
    assert buffer[0].obs.dtype == torch.uint8
    assert buffer[0].actions.shape == (10, 28)
    assert buffer[0].actions.dtype == torch.float32


def test_adroit_neural_policy():
    image_size = (3, 64, 64)
    policy = AdroitHandNeuralPolicy(image_size=image_size, action_dim=28, hidden_size=32, num_layers=2)
    assert isinstance(policy, AdroitHandPolicy)

    params = list(policy.parameters())
    assert params and all(p.requires_grad for p in params)

    obs = torch.rand(image_size)
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (28,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    # Backward gradient flow
    feat = obs.flatten()
    out = policy.actor(feat)[0]
    out.sum().backward()
    assert all(p.grad is not None for p in params)

    # Eval mode is deterministic
    policy.eval()
    assert torch.equal(policy(obs), policy(obs))


def test_adroit_policies_share_action_contract(adroit_env):
    env = TorchGymEnv(adroit_env)
    ref_action = env.sample_action()
    obs = torch.rand(3, 64, 64)

    policies = [
        AdroitHandRandomPolicy(),
        AdroitHandHeuristicPolicy(),
        AdroitHandNeuralPolicy(image_size=(3, 64, 64), action_dim=28),
    ]

    for policy in policies:
        policy.set_env(env)
        policy.reset()
        for step in range(3):
            action = policy(obs, step=step)
            name = type(policy).__name__
            assert isinstance(action, torch.Tensor), name
            assert action.shape == ref_action.shape, f"{name}: {action.shape} != {ref_action.shape}"
            assert action.dtype == ref_action.dtype, f"{name}: {action.dtype} != {ref_action.dtype}"
            assert env.action_space.contains(env._to_env_action(action)), name
