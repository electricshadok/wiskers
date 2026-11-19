import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

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
