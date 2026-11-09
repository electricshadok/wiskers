import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.transforms import EpisodeToTensor
from wiskers.envs import TorchGymEnv
from wiskers.envs.franka import (
    AVAILABLE_TASKS,
    FrankaKitchenHeuristicPolicy,
    FrankaKitchenNeuralPolicy,
    FrankaKitchenRandomPolicy,
    make_franka_kitchen_env,
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
def franka_env():
    env = make_franka_kitchen_env(tasks_to_complete=["microwave"], image_size=(64, 64))
    yield env
    env.close()


def test_make_franka_kitchen_env_observation_shape(franka_env):
    obs, info = franka_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8

    action = franka_env.action_space.sample()
    assert action.shape == (9,)
    next_obs, reward, terminated, truncated, info = franka_env.step(action)
    assert next_obs.shape == (64, 64, 3)
    assert next_obs.dtype == np.uint8
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_make_franka_kitchen_env_multiple_tasks():
    env = make_franka_kitchen_env(tasks_to_complete=["microwave", "kettle"], image_size=(32, 32))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32, 32, 3)
    env.close()


def test_make_franka_kitchen_env_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        make_franka_kitchen_env(tasks_to_complete=["non_existent_task"])


def test_franka_kitchen_available_tasks():
    assert "microwave" in AVAILABLE_TASKS
    assert "kettle" in AVAILABLE_TASKS
    assert "slide cabinet" in AVAILABLE_TASKS


def test_franka_random_policy(franka_env):
    torch_env = TorchGymEnv(franka_env)
    policy = FrankaKitchenRandomPolicy(env=torch_env)
    assert policy.action_dim == 9

    obs, _ = torch_env.reset()
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (9,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_franka_heuristic_policy(franka_env):
    torch_env = TorchGymEnv(franka_env)
    policy = FrankaKitchenHeuristicPolicy(momentum=0.8, noise_scale=0.2, env=torch_env)
    assert policy.action_dim == 9

    obs, _ = torch_env.reset()
    act1 = policy(obs, step=0)
    act2 = policy(obs, step=1)
    assert act1.shape == (9,)
    assert act2.shape == (9,)
    assert torch.all(act1 >= -1.0) and torch.all(act1 <= 1.0)
    assert torch.all(act2 >= -1.0) and torch.all(act2 <= 1.0)

    policy.reset()
    assert torch.all(policy.current_action == 0.0)


def test_franka_kitchen_collector_integration(franka_env):
    policy = FrankaKitchenHeuristicPolicy()
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=TorchGymEnv(franka_env),
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
    assert buffer[0].actions.shape == (10, 9)
    assert buffer[0].actions.dtype == torch.float32



def test_franka_neural_policy():
    image_size = (3, 64, 64)
    policy = FrankaKitchenNeuralPolicy(image_size=image_size, hidden_size=32, num_layers=2)

    params = list(policy.parameters())
    assert params and all(p.requires_grad for p in params)

    obs = torch.rand(image_size)
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (9,)
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


def test_franka_policies_share_action_contract(franka_env):
    env = TorchGymEnv(franka_env)
    ref_action = env.sample_action()
    obs = torch.rand(3, 64, 64)

    policies = [
        FrankaKitchenRandomPolicy(),
        FrankaKitchenHeuristicPolicy(),
        FrankaKitchenNeuralPolicy(image_size=(3, 64, 64)),
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
