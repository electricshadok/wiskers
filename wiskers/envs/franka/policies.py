"""
Policies for FrankaKitchen environment.
"""

from typing import Optional

import gymnasium as gym
import torch

from wiskers.common.policy import ContinuousPolicy


class FrankaKitchenPolicy(ContinuousPolicy):
    """Base policy for FrankaKitchen. Continuous 9D action space (7 arm joints + 2 gripper joints)."""

    action_dim: int = 9


class FrankaKitchenRandomPolicy(FrankaKitchenPolicy):
    """Uniform random exploration for FrankaKitchen."""

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        return self.sample_random()


class FrankaKitchenHeuristicPolicy(FrankaKitchenPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for continuous 9D control.

    Generates coherent, smooth reaching and grasping movements rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
        env: Optional Gymnasium environment instance.
    """

    def __init__(
        self,
        momentum: float = 0.85,
        noise_scale: float = 0.25,
        env: Optional[gym.Env] = None,
    ):
        super().__init__(env=env)
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action: torch.Tensor = torch.zeros(self.action_dim, dtype=torch.float32)

    def reset(self) -> None:
        self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        noise = torch.randn(self.action_dim, dtype=torch.float32) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = torch.clamp(action, -1.0, 1.0)
        return self.current_action.clone()
