import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.envs.utils import render_to_tensor


class DummyGymEnv(gym.Env):
    def __init__(self, shape=(120, 160, 3)):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 100, {}

    def step(self, action):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 200, 1.0, False, False, {}


def test_torch_gym_env_render_to_tensor_resize():
    frame = np.ones((120, 160, 3), dtype=np.uint8) * 100
    resized = render_to_tensor(frame, image_size=(64, 64))
    assert resized.shape == (64, 64, 3)
    assert resized.dtype == torch.uint8


def test_torch_gym_env_reset_and_step():
    env = DummyGymEnv(shape=(120, 160, 3))
    torch_env = TorchGymEnv(env)

    obs, info = torch_env.reset(seed=0)
    assert isinstance(obs, TensorDict)
    assert obs.batch_size == torch.Size([])
    assert obs["render"].shape == (120, 160, 3)
    assert obs["render"].dtype == torch.uint8

    action = torch.tensor([0.5, -0.5], dtype=torch.float32)
    next_obs, reward, term, trunc, info = torch_env.step(action)
    assert isinstance(next_obs, TensorDict)
    assert next_obs.batch_size == torch.Size([])
    assert next_obs["render"].shape == (120, 160, 3)
    assert next_obs["render"].dtype == torch.uint8
    assert reward == 1.0


def test_torch_gym_env_builds_action_spec_from_gym_spaces():
    box_env = TorchGymEnv(DummyGymEnv())
    assert isinstance(box_env.action_spec, BoundedActionSpec)
    assert box_env.action_spec.shape == (2,)
    assert box_env.action_spec.dtype == torch.float32
    assert box_env.action_spec.contains(box_env.sample_action())

    class IntBoxGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.int64)

    int_box_env = TorchGymEnv(IntBoxGymEnv())
    assert isinstance(int_box_env.action_spec, BoundedActionSpec)
    assert int_box_env.action_spec.shape == (3,)
    assert int_box_env.action_spec.dtype == torch.int64
    assert int_box_env.action_spec.contains(int_box_env.sample_action())

    class DiscreteGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Discrete(4)

    disc_env = TorchGymEnv(DiscreteGymEnv())
    assert isinstance(disc_env.action_spec, DiscreteActionSpec)
    assert disc_env.action_spec.n == 4
    assert disc_env.action_spec.contains(disc_env.sample_action())

    class UnsupportedGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.MultiDiscrete([2, 3])

    with pytest.raises(ValueError, match="Unsupported action space"):
        TorchGymEnv(UnsupportedGymEnv())
