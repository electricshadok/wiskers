"""
PyTorch wrapper around Gymnasium environments.
"""

from typing import Any, Self

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import ActionSpec, BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.utils import render_to_tensor


class GymEnv:
    """
    Adapter between Gymnasium and PyTorch.

    Translates PyTorch Tensor actions to Gymnasium environment inputs and wraps
    observations into TensorDicts.

    Args:
        env (gym.Env): The underlying Gymnasium environment.
    """

    def __init__(self, env: gym.Env) -> None:
        self._gym_env = env
        self.action_spec = self._action_spec()

    def close(self) -> None:
        self._gym_env.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _process_obs(self, raw_obs: Any) -> TensorDict:
        """Convert a raw Gymnasium observation into a TensorDict with batch_size=[]."""
        return TensorDict({"render": render_to_tensor(raw_obs)}, batch_size=[])

    def reset(
        self,
        seed: int,
        options: dict[str, Any] | None = None,
    ) -> tuple[TensorDict, dict[str, Any]]:
        raw_obs, info = self._gym_env.reset(seed=seed, options=options)
        return self._process_obs(raw_obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[TensorDict, float, bool, bool, dict[str, Any]]:
        gym_action = self._to_gym_action(action)
        raw_obs, reward, terminated, truncated, info = self._gym_env.step(gym_action)
        return self._process_obs(raw_obs), float(reward), terminated, truncated, info

    def _action_spec(self) -> ActionSpec:
        """Translate ``self._gym_env.action_space`` into a PyTorch ``ActionSpec``."""
        space = self._gym_env.action_space
        if isinstance(space, gym.spaces.Discrete):
            return DiscreteActionSpec(n=int(space.n))
        if isinstance(space, gym.spaces.Box):
            if not np.issubdtype(space.dtype, np.floating):
                raise ValueError(f"Unsupported Box action dtype: {space.dtype}")
            low = torch.as_tensor(space.low, dtype=torch.float32)
            high = torch.as_tensor(space.high, dtype=torch.float32)
            return BoundedActionSpec(low=low, high=high)
        raise ValueError(f"Unsupported action space: {type(space)}")

    def _to_gym_action(self, action: torch.Tensor) -> int | np.ndarray:
        action_cpu = action.detach().cpu()
        if isinstance(self.action_spec, DiscreteActionSpec):
            return int(action_cpu.item())
        if isinstance(self.action_spec, BoundedActionSpec):
            return action_cpu.to(dtype=self.action_spec.dtype).numpy()
        raise ValueError(f"Unsupported action spec: {type(self.action_spec)}")
