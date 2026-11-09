import random
from typing import Optional, Sequence

import torch

from wiskers.common.heads import ContinuousActor
from wiskers.common.policy import ContinuousPolicy
from wiskers.envs.torch_env import TorchGymEnv


class CarRacingPolicy(ContinuousPolicy):
    """Base policy for CarRacing. Continuous action space (steering, gas, brake)."""

    action_dim: int = 3


class CarRacingRandomPolicy(CarRacingPolicy):
    """Uniform random exploration for CarRacing."""

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        return self.sample_random()


class CarRacingHeuristicPolicy(CarRacingPolicy):
    """
    OU-style mean-reverting random steering keeps the car on track.

    Internal state:
        steering: smoothed steering angle in [-1, 1], reset to 0.0 at episode start.
    """

    def __init__(self, env: Optional[TorchGymEnv] = None):
        super().__init__(env=env)
        self.steering = 0.0

    def reset(self) -> None:
        self.steering = 0.0

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        self.steering = self.steering * 0.8 + random.uniform(-0.2, 0.2)
        self.steering = max(-1.0, min(1.0, self.steering))
        gas = random.uniform(0.4, 0.8)
        brake = random.uniform(0.0, 0.05)
        return torch.tensor([self.steering, gas, brake], dtype=torch.float32)


class CarRacingNeuralPolicy(CarRacingPolicy):
    """
    Minimal trainable policy for CarRacing: flattened pixels → MLP → squashed Gaussian.

    Samples in train mode and uses the mean in eval mode. The actor outputs values in
    (-1, 1); gas and brake are rescaled to [0, 1] to match the action space.

    Args:
        image_size: Transformed observation shape ``(C, H, W)``.
        hidden_size: MLP hidden width.
        num_layers: Number of MLP hidden layers.
        env: Optional TorchGymEnv environment instance.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        hidden_size: int = 64,
        num_layers: int = 2,
        env: Optional[TorchGymEnv] = None,
    ):
        super().__init__(env=env)
        c, h, w = image_size
        self.actor = ContinuousActor(
            feat_size=c * h * w,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        feat = obs.flatten(start_dim=-3)
        action = self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
        steering = action[..., :1]
        gas_brake = (action[..., 1:] + 1.0) / 2.0
        return torch.cat([steering, gas_brake], dim=-1)
