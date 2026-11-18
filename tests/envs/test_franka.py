import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.franka_kitchen import (
    AVAILABLE_TASKS,
    FrankaKitchenEnv,
    FrankaKitchenHeuristicPolicy,
    FrankaKitchenNeuralPolicy,
    FrankaKitchenRandomPolicy,
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
    env = FrankaKitchenEnv(image_size=(64, 64), tasks_to_complete=["microwave"])
    yield env
    env.close()


def test_franka_kitchen_env_observation_shape():
    env = FrankaKitchenEnv(tasks_to_complete=["microwave"])
    obs, info = env.reset(seed=42)
    assert isinstance(obs, TensorDict)
    assert obs.batch_size == torch.Size([])
    assert obs["render"].shape == (480, 480, 3)
    assert obs["render"].dtype == torch.uint8
    assert obs["state:robot_qpos"].shape == (9,)
    assert obs["state:robot_qpos"].dtype == torch.float32
    assert obs["state:robot_qvel"].shape == (9,)
    assert obs["state:robot_qvel"].dtype == torch.float32
    assert obs["state:obj_qpos"].shape == (21,)
    assert obs["state:obj_qpos"].dtype == torch.float32
    assert obs["state:obj_qvel"].shape == (20,)
    assert obs["state:obj_qvel"].dtype == torch.float32

    action = env.sample_action()
    assert action.shape == (9,)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(next_obs, TensorDict)
    assert next_obs.batch_size == torch.Size([])
    assert next_obs["render"].shape == (480, 480, 3)
    assert next_obs["render"].dtype == torch.uint8
    assert next_obs["state:robot_qpos"].shape == (9,)
    assert next_obs["state:robot_qpos"].dtype == torch.float32
    assert next_obs["state:robot_qvel"].shape == (9,)
    assert next_obs["state:robot_qvel"].dtype == torch.float32
    assert next_obs["state:obj_qpos"].shape == (21,)
    assert next_obs["state:obj_qpos"].dtype == torch.float32
    assert next_obs["state:obj_qvel"].shape == (20,)
    assert next_obs["state:obj_qvel"].dtype == torch.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_franka_kitchen_state_dimensions_match_farama_spec(franka_env):
    assert franka_env.env.observation_space["observation"].shape == (59,)

    raw_obs, _ = franka_env.env.reset(seed=42)
    expected_raw = torch.from_numpy(np.ascontiguousarray(raw_obs["observation"], dtype=np.float32))

    obs, _ = franka_env.reset(seed=42)
    expected_dims = {
        "state:robot_qpos": 9,
        "state:robot_qvel": 9,
        "state:obj_qpos": 21,
        "state:obj_qvel": 20,
    }
    assert [k for k in obs.keys() if k.startswith("state:")] == list(expected_dims.keys())
    for key, dim in expected_dims.items():
        assert obs[key].shape == (dim,)
        assert obs[key].dtype == torch.float32

    assert sum(obs[key].shape[-1] for key in expected_dims) == 59
    reconstructed = torch.cat([obs[key] for key in expected_dims], dim=-1)
    assert torch.equal(reconstructed, expected_raw)


def test_franka_kitchen_env_multiple_tasks():
    env = FrankaKitchenEnv(tasks_to_complete=["microwave", "kettle"])
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, TensorDict)
    assert obs["render"].shape == (480, 480, 3)
    env.close()


def test_franka_kitchen_env_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        FrankaKitchenEnv(tasks_to_complete=["non_existent_task"])


def test_franka_kitchen_available_tasks():
    assert "microwave" in AVAILABLE_TASKS
    assert "kettle" in AVAILABLE_TASKS
    assert "slide cabinet" in AVAILABLE_TASKS


def test_franka_random_policy(franka_env):
    policy = FrankaKitchenRandomPolicy()
    assert policy.action_dim == 9

    obs, _ = franka_env.reset()
    action = policy(obs["render"], step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (9,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_franka_heuristic_policy(franka_env):
    policy = FrankaKitchenHeuristicPolicy(momentum=0.8, noise_scale=0.2)
    assert policy.action_dim == 9

    obs, _ = franka_env.reset()
    act1 = policy(obs["render"], step=0)
    act2 = policy(obs["render"], step=1)
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
        env=franka_env,
        controller=policy,
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
    assert buffer[0].obs["render"].shape == (10, 64, 64, 3)
    assert buffer[0].obs["render"].dtype == torch.uint8
    assert buffer[0].obs["state:robot_qpos"].shape == (10, 9)
    assert buffer[0].obs["state:robot_qpos"].dtype == torch.float32
    assert buffer[0].obs["state:robot_qvel"].shape == (10, 9)
    assert buffer[0].obs["state:robot_qvel"].dtype == torch.float32
    assert buffer[0].obs["state:obj_qpos"].shape == (10, 21)
    assert buffer[0].obs["state:obj_qpos"].dtype == torch.float32
    assert buffer[0].obs["state:obj_qvel"].shape == (10, 20)
    assert buffer[0].obs["state:obj_qvel"].dtype == torch.float32
    assert buffer[0].action.shape == (10, 9)
    assert buffer[0].action.dtype == torch.float32


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
    ref_action = franka_env.sample_action()
    obs = torch.rand(3, 64, 64)

    policies = [
        FrankaKitchenRandomPolicy(),
        FrankaKitchenHeuristicPolicy(),
        FrankaKitchenNeuralPolicy(image_size=(3, 64, 64)),
    ]

    for policy in policies:
        policy.reset()
        for step in range(3):
            action = policy(obs, step=step)
            name = type(policy).__name__
            assert isinstance(action, torch.Tensor), name
            assert action.shape == ref_action.shape, f"{name}: {action.shape} != {ref_action.shape}"
            assert action.dtype == ref_action.dtype, f"{name}: {action.dtype} != {ref_action.dtype}"
            assert franka_env.action_space.contains(franka_env._to_env_action(action)), name
