from typing import Optional

import gymnasium as gym
import numpy as np

from wiskers.common.policy import ContinuousPolicy


class CarRacingPolicy(ContinuousPolicy):
    """Base policy for CarRacing. Continuous action space (steering, gas, brake)."""

    action_dim: int = 3


class CarRacingRandomPolicy(CarRacingPolicy):
    """Uniform random exploration for CarRacing."""

    def forward(self, obs: np.ndarray, step: int = 0) -> np.ndarray:
        return self.sample_random()


class CarRacingHeuristicPolicy(CarRacingPolicy):
    """
    OU-style mean-reverting random steering keeps the car on track.

    Internal state:
        steering: smoothed steering angle in [-1, 1], reset to 0.0 at episode start.
    """

    def __init__(self, env: Optional[gym.Env] = None):
        super().__init__(env=env)
        self.steering = 0.0

    def reset(self) -> None:
        self.steering = 0.0

    def forward(self, obs: np.ndarray, step: int = 0) -> np.ndarray:
        self.steering = self.steering * 0.8 + np.random.uniform(-0.2, 0.2)
        self.steering = float(np.clip(self.steering, -1.0, 1.0))
        gas = float(np.random.uniform(0.4, 0.8))
        brake = float(np.random.uniform(0.0, 0.05))
        return np.array([self.steering, gas, brake], dtype=np.float32)
