from typing import Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn


class CarRacingPolicy(nn.Module):
    """Base policy for CarRacing. Continuous action space (steering, gas, brake)."""

    action_dim: int = 3
    action_type: str = "continuous"

    def reset(self) -> None:
        """Reset internal rollout state at episode start."""
        pass

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        """Return 3D continuous action vector [steering, gas, brake]."""
        raise NotImplementedError


class CarRacingRandomPolicy(CarRacingPolicy):
    """Uniform random exploration for CarRacing."""

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        if env is not None:
            return np.asarray(env.action_space.sample(), dtype=np.float32)
        steering = float(np.random.uniform(-1.0, 1.0))
        gas = float(np.random.uniform(0.0, 1.0))
        brake = float(np.random.uniform(0.0, 1.0))
        return np.array([steering, gas, brake], dtype=np.float32)


class CarRacingHeuristicPolicy(CarRacingPolicy):
    """
    OU-style mean-reverting random steering keeps the car on track.

    Internal state:
        steering: smoothed steering angle in [-1, 1], reset to 0.0 at episode start.
    """

    def __init__(self):
        super().__init__()
        self.steering = 0.0

    def reset(self) -> None:
        self.steering = 0.0

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> np.ndarray:
        self.steering = self.steering * 0.8 + np.random.uniform(-0.2, 0.2)
        self.steering = float(np.clip(self.steering, -1.0, 1.0))
        gas = float(np.random.uniform(0.4, 0.8))
        brake = float(np.random.uniform(0.0, 0.05))
        return np.array([self.steering, gas, brake], dtype=np.float32)
