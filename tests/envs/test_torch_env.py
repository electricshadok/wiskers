import gymnasium as gym
import numpy as np
import torch

from wiskers.envs import TorchGymEnv


class DummyGymEnv(gym.Env):
    def __init__(self, shape=(120, 160, 3)):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 100, {}

    def step(self, action):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 200, 1.0, False, False, {}


def test_torch_gym_env_resize():
    env = DummyGymEnv(shape=(120, 160, 3))
    torch_env = TorchGymEnv(env, image_size=(64, 64))

    assert torch_env.observation_space.shape == (64, 64, 3)

    obs, info = torch_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == torch.uint8

    action = torch.tensor([0.5, -0.5], dtype=torch.float32)
    next_obs, reward, term, trunc, info = torch_env.step(action)
    assert isinstance(next_obs, torch.Tensor)
    assert next_obs.shape == (64, 64, 3)
    assert next_obs.dtype == torch.uint8
    assert reward == 1.0


def test_torch_gym_env_no_resize():
    env = DummyGymEnv(shape=(120, 160, 3))
    torch_env = TorchGymEnv(env, image_size=None)

    assert torch_env.observation_space.shape == (120, 160, 3)

    obs, _ = torch_env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (120, 160, 3)

