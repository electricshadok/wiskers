import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch.nn as nn


class BreakoutPolicy(nn.Module):
    """Base policy for ALE/Breakout. Discrete action space (0..3)."""

    action_dim: int = 1
    action_type: str = "discrete"

    def reset(self) -> None:
        """Reset internal state at episode start."""
        pass

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> int:
        """Return a discrete action index (0..3)."""
        raise NotImplementedError


class BreakoutRandomPolicy(BreakoutPolicy):
    """Uniform random exploration for Breakout with periodic FIRE to unstick the ball."""

    def __init__(self, fire_frequency: int = 30):
        super().__init__()
        self.fire_frequency = fire_frequency

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> int:
        if step % self.fire_frequency == 0:
            return 1  # FIRE
        if env is not None:
            return int(env.action_space.sample())
        return random.randint(0, 3)


class BreakoutHeuristicPolicy(BreakoutPolicy):
    """
    Heuristic policy that tracks ball movement using ALE RAM state.

    RAM addresses for Breakout:
        72  : paddle x-coordinate
        99  : ball x-coordinate
        101 : ball y-coordinate
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, obs: np.ndarray, env: Optional[gym.Env] = None, step: int = 0) -> int:
        if random.random() < self.epsilon or env is None:
            return 1 if step == 0 else (int(env.action_space.sample()) if env is not None else 0)

        try:
            ale = getattr(env.unwrapped, "ale", None)
            if ale is None:
                return int(env.action_space.sample())

            ram = ale.getRAM()
            paddle_x = int(ram[72])
            ball_x = int(ram[99])
            ball_y = int(ram[101])

            # If ball is waiting to be served or just reset, press FIRE
            if ball_y == 0 or step == 0:
                return 1  # FIRE

            # Track the ball with the paddle
            if paddle_x < ball_x - 2:
                return 2  # RIGHT
            elif paddle_x > ball_x + 2:
                return 3  # LEFT
            else:
                return random.choice([0, 2, 3])
        except Exception:
            return int(env.action_space.sample()) if env is not None else 0
