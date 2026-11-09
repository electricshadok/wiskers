import random
from typing import Optional, Sequence

import torch

from wiskers.common.heads import CategoricalActor
from wiskers.common.policy import DiscretePolicy
from wiskers.envs.torch_env import TorchGymEnv


class BreakoutPolicy(DiscretePolicy):
    """Base policy for ALE/Breakout. Discrete action space (0..3)."""
    pass


class BreakoutRandomPolicy(BreakoutPolicy):
    """Uniform random exploration for Breakout with periodic FIRE to unstick the ball."""

    def __init__(self, fire_frequency: int = 30, env: Optional[TorchGymEnv] = None):
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

    def __init__(self, epsilon: float = 0.1, env: Optional[TorchGymEnv] = None):
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


class BreakoutNeuralPolicy(BreakoutPolicy):
    """
    Minimal trainable policy for Breakout: flattened pixels → MLP → categorical logits.

    Samples from the categorical distribution in train mode and takes the argmax in eval mode.

    Args:
        image_size: Transformed observation shape ``(C, H, W)``.
        hidden_size: MLP hidden width.
        num_layers: Number of MLP hidden layers.
        env: Optional TorchGymEnv environment instance.
    """

    num_actions: int = 4

    def __init__(
        self,
        image_size: Sequence[int],
        hidden_size: int = 64,
        num_layers: int = 2,
        env: Optional[TorchGymEnv] = None,
    ):
        super().__init__(env=env)
        c, h, w = image_size
        self.actor = CategoricalActor(
            feat_size=c * h * w,
            num_actions=self.num_actions,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        feat = obs.flatten(start_dim=-3)
        if self.training:
            return self.actor.dist(feat).sample()
        return self.actor(feat).argmax(dim=-1)
