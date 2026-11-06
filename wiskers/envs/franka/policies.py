"""
Policies for FrankaKitchen environment.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn


class FrankaKitchenPolicy(nn.Module):
    """Base policy for FrankaKitchen. Continuous 9D action space (7 arm joints + 2 gripper joints)."""

    action_dim: int = 9
    action_type: str = "continuous"

    def reset(self) -> None:
        """Reset internal rollout state at episode start."""
        pass

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        """Return 9D continuous action vector in [-1.0, 1.0]."""
        raise NotImplementedError


class FrankaKitchenRandomPolicy(FrankaKitchenPolicy):
    """Uniform random exploration for FrankaKitchen."""

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        if env is not None:
            return np.asarray(env.action_space.sample(), dtype=np.float32)
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)


class FrankaKitchenHeuristicPolicy(FrankaKitchenPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for continuous 9D control.

    Generates coherent, smooth reaching and grasping movements rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
    """

    def __init__(self, momentum: float = 0.85, noise_scale: float = 0.25):
        super().__init__()
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        noise = np.random.randn(self.action_dim).astype(np.float32) * self.noise_scale
        self.current_action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = np.clip(self.current_action, -1.0, 1.0)
        return self.current_action.copy()

