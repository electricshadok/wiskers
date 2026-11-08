import random
from typing import Optional

import gymnasium as gym
import torch

from wiskers.common.policy import DiscretePolicy


class BreakoutPolicy(DiscretePolicy):
    """Base policy for ALE/Breakout. Discrete action space (0..3)."""
    pass


class BreakoutRandomPolicy(BreakoutPolicy):
    """Uniform random exploration for Breakout with periodic FIRE to unstick the ball."""

    def __init__(self, fire_frequency: int = 30, env: Optional[gym.Env] = None):
        super().__init__(env=env)
        self.fire_frequency = fire_frequency

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        if step % self.fire_frequency == 0:
            return torch.tensor(1, dtype=torch.long)  # FIRE
        return self.sample_random()


class BreakoutHeuristicPolicy(BreakoutPolicy):
    """
    Heuristic policy that tracks ball movement using ALE RAM state.

    RAM addresses for Breakout:
        72  : paddle x-coordinate
        99  : ball x-coordinate
        101 : ball y-coordinate
    """

    def __init__(self, epsilon: float = 0.1, env: Optional[gym.Env] = None):
        super().__init__(env=env)
        self.epsilon = epsilon

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        if random.random() < self.epsilon or self.env is None:
            return torch.tensor(1, dtype=torch.long) if step == 0 else self.sample_random()

        try:
            ale = getattr(self.env.unwrapped, "ale", None)
            if ale is None:
                return self.sample_random()

            ram = ale.getRAM()
            paddle_x = int(ram[72])
            ball_x = int(ram[99])
            ball_y = int(ram[101])

            # If ball is waiting to be served or just reset, press FIRE
            if ball_y == 0 or step == 0:
                return torch.tensor(1, dtype=torch.long)  # FIRE

            # Track the ball with the paddle
            if paddle_x < ball_x - 2:
                return torch.tensor(2, dtype=torch.long)  # RIGHT
            elif paddle_x > ball_x + 2:
                return torch.tensor(3, dtype=torch.long)  # LEFT
            else:
                return torch.tensor(random.choice([0, 2, 3]), dtype=torch.long)
        except Exception:
            return self.sample_random()
