"""
Policies for AdroitHand dexterous manipulation environments.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn


class AdroitHandPolicy(nn.Module):
    """Base policy for AdroitHand. Continuous action space (24..30 DoF depending on task)."""

    action_dim: int = 28
    action_type: str = "continuous"

    def __init__(self, action_dim: int = 28):
        super().__init__()
        self.action_dim = action_dim

    def reset(self) -> None:
        """Reset internal rollout state at episode start."""
        pass

    def _get_action_dim(self, env: Optional[gym.Env] = None) -> int:
        if env is not None and hasattr(env, "action_space") and env.action_space.shape:
            return int(env.action_space.shape[0])
        return self.action_dim

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        """Return continuous action vector in [-1.0, 1.0]."""
        raise NotImplementedError


class AdroitHandRandomPolicy(AdroitHandPolicy):
    """Uniform random exploration for AdroitHand."""

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        if env is not None:
            return np.asarray(env.action_space.sample(), dtype=np.float32)
        dim = self._get_action_dim(env)
        return np.random.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)


class AdroitHandHeuristicPolicy(AdroitHandPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for AdroitHand control.

    Generates coherent reaching and finger-articulation trajectories rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
        action_dim: Default action dimension when ``env`` is not provided (28 for door).
    """

    def __init__(self, momentum: float = 0.85, noise_scale: float = 0.25, action_dim: int = 28):
        super().__init__(action_dim=action_dim)
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        dim = self._get_action_dim(env)
        if self.current_action.shape[0] != dim:
            self.current_action = np.zeros(dim, dtype=np.float32)
        noise = np.random.randn(dim).astype(np.float32) * self.noise_scale
        self.current_action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = np.clip(self.current_action, -1.0, 1.0)
        return self.current_action.copy()

