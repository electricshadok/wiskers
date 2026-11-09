"""
PyTorch wrapper around Gymnasium environments.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch


class TorchGymEnv(gym.Wrapper):
    """
    Wrapper around a Gymnasium environment that converts NumPy observations
    to CPU ``torch.uint8`` tensors and ``torch.Tensor`` actions back to NumPy/int
    at the environment boundary.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_obs, info = self.env.reset(seed=seed, options=options)
        return torch.from_numpy(np.ascontiguousarray(raw_obs)), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        env_action = self._to_env_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = torch.from_numpy(np.ascontiguousarray(raw_obs))
        return obs, float(reward), terminated, truncated, info

    def sample_action(self) -> torch.Tensor:
        """Sample a random action from ``self.env.action_space`` as a ``torch.Tensor``."""
        sample = self.env.action_space.sample()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return torch.tensor(int(sample), dtype=torch.long)
        return torch.from_numpy(np.asarray(sample, dtype=np.float32))

    def _to_env_action(self, action: torch.Tensor) -> int | np.ndarray:
        action_cpu = action.detach().cpu()
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return int(action_cpu.item())
        return action_cpu.to(dtype=torch.float32).numpy()

