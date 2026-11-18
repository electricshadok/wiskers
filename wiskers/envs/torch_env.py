"""
PyTorch wrapper around Gymnasium environments.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.utils import render_to_tensor


class TorchGymEnv:
    """
    Adapter between Gymnasium and PyTorch.

    Translates PyTorch Tensor actions to Gymnasium environment inputs and wraps
    observations into TensorDicts.

    Args:
        env (gym.Env): The underlying Gymnasium environment.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env

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

    def _process_obs(self, raw_obs: Any) -> TensorDict:
        """Convert a raw Gymnasium observation into a TensorDict with batch_size=[]."""
        return TensorDict({"render": render_to_tensor(raw_obs)}, batch_size=[])

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
