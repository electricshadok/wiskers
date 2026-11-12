"""
PyTorch wrapper around Gymnasium environments.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class TorchGymEnv(gym.Wrapper):
    """
    Adapter between Gymnasium and PyTorch.

    Converts raw NumPy observations to PyTorch uint8 CPU tensors, performs
    spatial downsampling directly in PyTorch via F.interpolate(mode='area'),
    and translates torch.Tensor actions back to environment format.

    Args:
        env (gym.Env): The underlying Gymnasium environment.
        image_size (Optional[tuple[int, int]]): Target (height, width) to resize image frames to.
                                                If None, preserves the environment's native resolution.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(env)
        self.image_size = tuple(image_size) if image_size is not None else None

        obs_space = getattr(self.env, "observation_space", None)
        if self.image_size is not None and obs_space is not None and hasattr(obs_space, "shape"):
            old_shape = obs_space.shape
            if len(old_shape) >= 2:
                new_shape = self.image_size + old_shape[2:]
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=new_shape, dtype=np.uint8
                )

    def _process_obs(self, raw_obs: np.ndarray) -> torch.Tensor:
        obs = torch.from_numpy(np.ascontiguousarray(raw_obs))
        if self.image_size is not None and obs.ndim >= 2:
            if obs.shape[:2] == self.image_size:
                return obs
            if obs.ndim == 3:
                t = obs.permute(2, 0, 1).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.image_size, mode="area")
                return resized.squeeze(0).permute(1, 2, 0).to(torch.uint8)
            elif obs.ndim == 2:
                t = obs.unsqueeze(0).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.image_size, mode="area")
                return resized.squeeze(0).squeeze(0).to(torch.uint8)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(raw_obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        env_action = self._to_env_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        return self._process_obs(raw_obs), float(reward), terminated, truncated, info

    def sample_action(self) -> torch.Tensor:
        """Sample a random action from self.env.action_space as a torch.Tensor."""
        sample = self.env.action_space.sample()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return torch.tensor(int(sample), dtype=torch.long)
        return torch.from_numpy(np.asarray(sample, dtype=np.float32))

    def _to_env_action(self, action: torch.Tensor) -> int | np.ndarray:
        action_cpu = action.detach().cpu()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(action_cpu.item())
        return action_cpu.to(dtype=torch.float32).numpy()
