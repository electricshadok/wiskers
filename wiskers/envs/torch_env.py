"""
PyTorch wrapper around Gymnasium environments.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict


class TorchGymEnv:
    """
    Adapter between Gymnasium and PyTorch.

    Converts raw NumPy observations to PyTorch uint8 CPU tensors, performs
    spatial downsampling directly in PyTorch via F.interpolate(mode='area'),
    and translates torch.Tensor actions back to environment format.

    Args:
        env (gym.Env): The underlying Gymnasium environment.
        image_size (tuple[int, int] | None): Target (height, width) to resize image frames to.
                                             If None, preserves the environment's native resolution.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.env = env
        self.image_size = tuple(image_size) if image_size is not None else None

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def unwrapped(self) -> gym.Env:
        return self.env.unwrapped

    @property
    def spec(self) -> Any:
        return self.env.spec

    def close(self) -> None:
        self.env.close()

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        obs = torch.from_numpy(np.ascontiguousarray(raw_obs))
        if self.image_size is not None and obs.ndim >= 2 and obs.shape[:2] != self.image_size:
            if obs.ndim == 3:
                t = obs.permute(2, 0, 1).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.image_size, mode="area")
                obs = resized.squeeze(0).permute(1, 2, 0).to(torch.uint8)
            elif obs.ndim == 2:
                t = obs.unsqueeze(0).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.image_size, mode="area")
                obs = resized.squeeze(0).squeeze(0).to(torch.uint8)
        return TensorDict({"render": obs}, batch_size=[])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TensorDict, dict[str, Any]]:
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(raw_obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[TensorDict, float, bool, bool, dict[str, Any]]:
        env_action = self._to_env_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        return self._process_obs(raw_obs), float(reward), terminated, truncated, info

    def sample_action(self) -> torch.Tensor:
        """Sample a random action from self.env.action_space as a torch.Tensor."""
        sample = self.env.action_space.sample()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return torch.tensor([int(sample)], dtype=torch.long)
        return torch.from_numpy(np.asarray(sample, dtype=np.float32))

    def _to_env_action(self, action: torch.Tensor) -> int | np.ndarray:
        action_cpu = action.detach().cpu()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(action_cpu.item())
        return action_cpu.to(dtype=torch.float32).numpy()
