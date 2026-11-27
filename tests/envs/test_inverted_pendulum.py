import typing

import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.action_spec import BoundedActionSpec
from wiskers.envs.inverted_pendulum.env import InvertedPendulumEnv
from wiskers.envs.inverted_pendulum.policies import InvertedPendulumNeuralPolicy
from wiskers.envs.utils import ResizeUint8
from wiskers.pipeline.config import load_config


@pytest.fixture(autouse=True)
def mock_mujoco_renderer(monkeypatch):
    """Bypass OpenGL/GLFW window creation during unit tests while keeping real MuJoCo physics."""
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )


@pytest.fixture
def inverted_pendulum_env():
    env = InvertedPendulumEnv(obs_fields={"render": ResizeUint8((64, 64))})
    yield env
    env.close()


def test_inverted_pendulum_env_observation_shape():
    env = InvertedPendulumEnv()
    try:
        assert isinstance(env.action_spec, BoundedActionSpec)
        assert torch.equal(env.action_spec.low, torch.tensor([-3.0], dtype=torch.float32))
        assert torch.equal(env.action_spec.high, torch.tensor([3.0], dtype=torch.float32))

        obs, info = env.reset(seed=42)
        assert isinstance(obs, TensorDict)
        assert isinstance(info, dict)
        assert obs.batch_size == torch.Size([])
        assert obs["render"].shape == (480, 480, 3)
        assert obs["render"].dtype == torch.uint8
        assert obs["state:qpos"].shape == (2,)
        assert obs["state:qpos"].dtype == torch.float32
        assert obs["state:qvel"].shape == (2,)
        assert obs["state:qvel"].dtype == torch.float32

        action = env.action_spec.sample()
        assert action.shape == (1,)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert isinstance(next_obs, TensorDict)
        assert next_obs.batch_size == torch.Size([])
        assert next_obs["render"].shape == (480, 480, 3)
        assert next_obs["render"].dtype == torch.uint8
        assert next_obs["state:qpos"].shape == (2,)
        assert next_obs["state:qpos"].dtype == torch.float32
        assert next_obs["state:qvel"].shape == (2,)
        assert next_obs["state:qvel"].dtype == torch.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()


def test_inverted_pendulum_state_dimensions_match_gymnasium_spec(inverted_pendulum_env):
    assert inverted_pendulum_env._gym_env.observation_space.shape == (4,)
    unwrapped: typing.Any = inverted_pendulum_env._gym_env.unwrapped
    assert unwrapped.observation_structure == {"qpos": 2, "qvel": 2}

    raw_obs, _ = inverted_pendulum_env._gym_env.reset(seed=42)
    expected_raw = torch.from_numpy(np.ascontiguousarray(raw_obs, dtype=np.float32))

    obs, _ = inverted_pendulum_env.reset(seed=42)
    expected_dims = {
        "state:qpos": 2,
        "state:qvel": 2,
    }
    assert [k for k in obs.keys() if k.startswith("state:")] == list(expected_dims.keys())
    for key, dim in expected_dims.items():
        assert obs[key].shape == (dim,)
        assert obs[key].dtype == torch.float32

    reconstructed = torch.cat([obs[key] for key in expected_dims], dim=-1)
    assert torch.equal(reconstructed, expected_raw)


def test_inverted_pendulum_reset_noise_scale():
    env = InvertedPendulumEnv(reset_noise_scale=0.0)
    try:
        obs, _ = env.reset(seed=42)
        assert torch.all(obs["state:qpos"] == 0.0)
        assert torch.all(obs["state:qvel"] == 0.0)
    finally:
        env.close()

    with pytest.raises(ValueError, match="reset_noise_scale must be non-negative"):
        InvertedPendulumEnv(reset_noise_scale=-0.1)


def test_inverted_pendulum_neural_policy():
    image_size = (3, 64, 64)
    policy = InvertedPendulumNeuralPolicy(image_size=image_size, hidden_size=32, num_layers=2)

    params = list(policy.parameters())
    assert params and all(p.requires_grad for p in params)

    obs = TensorDict({"render": torch.rand(image_size)}, batch_size=[])
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (1,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -3.0) and torch.all(action <= 3.0)

    # Backward gradient flow through policy(obs)
    policy.zero_grad()
    policy.train()
    policy(obs, step=0).sum().backward()
    assert all(p.grad is not None for p in params)

    # Eval mode is deterministic
    policy.eval()
    assert torch.equal(policy(obs), policy(obs))


def test_inverted_pendulum_policies_share_action_contract(inverted_pendulum_env):
    ref_action = inverted_pendulum_env.action_spec.sample()
    obs = TensorDict(
        {
            "render": torch.rand(3, 64, 64),
            "state:qpos": torch.zeros(2, dtype=torch.float32),
            "state:qvel": torch.zeros(2, dtype=torch.float32),
        },
        batch_size=[],
    )

    policies = [
        InvertedPendulumNeuralPolicy(image_size=(3, 64, 64)),
    ]

    for policy in policies:
        assert policy.action_spec == inverted_pendulum_env.action_spec, type(policy).__name__
        policy.reset()
        for step in range(3):
            action = policy(obs, step=step)
            name = type(policy).__name__
            assert isinstance(action, torch.Tensor), name
            assert action.shape == ref_action.shape, f"{name}: {action.shape} != {ref_action.shape}"
            assert action.dtype == ref_action.dtype, f"{name}: {action.dtype} != {ref_action.dtype}"
            assert inverted_pendulum_env.action_spec.contains(action), name


def test_inverted_pendulum_collector_integration(inverted_pendulum_env):
    policy = InvertedPendulumNeuralPolicy(image_size=(3, 64, 64))
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=inverted_pendulum_env,
        controller=policy,
        transform=transform,
        num_episodes=1,
        min_steps=4,
        max_steps=10,
        seed=0,
        verbose=False,
    )
    buffer = ReplayBuffer(capacity_episodes=5)
    collected_steps = collector.collect(buffer)

    assert 4 <= collected_steps <= 10
    assert buffer.num_episodes == 1
    assert buffer.num_steps == collected_steps
    assert buffer[0].obs["render"].shape == (collected_steps, 64, 64, 3)
    assert buffer[0].obs["render"].dtype == torch.uint8
    assert buffer[0].obs["state:qpos"].shape == (collected_steps, 2)
    assert buffer[0].obs["state:qpos"].dtype == torch.float32
    assert buffer[0].obs["state:qvel"].shape == (collected_steps, 2)
    assert buffer[0].obs["state:qvel"].dtype == torch.float32
    assert buffer[0].action.shape == (collected_steps, 1)
    assert buffer[0].action.dtype == torch.float32


def test_inverted_pendulum_dataset_config():
    cfg = load_config("dataset/inverted_pendulum")
    assert cfg.data_module.collector.env._target_ == "wiskers.envs.inverted_pendulum.env.InvertedPendulumEnv"
    assert (
        cfg.data_module.collector.controller._target_
        == "wiskers.envs.inverted_pendulum.policies.InvertedPendulumNeuralPolicy"
    )
    assert list(cfg.data_module.collector.controller.image_size) == [3, 64, 64]
    assert cfg.data_module.sampler.num_steps == 4
    assert cfg.data_module.collector.min_steps == 4
