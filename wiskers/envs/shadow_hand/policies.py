"""
Policies for Shadow Dexterous Hand manipulation environments.
"""

from typing import Sequence

import gymnasium as gym
import torch
from tensordict import TensorDict

from wiskers.nn.heads import ContinuousActor
from wiskers.policies.base import ContinuousPolicy


class ShadowHandPolicy(ContinuousPolicy):
    """Base policy for ShadowHand. Continuous 20D action space across all tasks."""

    action_dim: int = 20


class ShadowHandRandomPolicy(ShadowHandPolicy):
    """Uniform random exploration for ShadowHand."""

    def __init__(self, action_space: gym.spaces.Box | None = None) -> None:
        super().__init__()
        self.action_space = action_space or gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=float
        )

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        return torch.from_numpy(self.action_space.sample()).float()


class ShadowHandHeuristicPolicy(ShadowHandPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for ShadowHand control.

    Generates coherent wrist and finger articulation trajectories rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
    """

    def __init__(
        self,
        momentum: float = 0.85,
        noise_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)

    def reset(self) -> None:
        self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        noise = torch.randn(self.action_dim, dtype=torch.float32) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = torch.clamp(action, -1.0, 1.0)
        return self.current_action.clone()


class ShadowHandNeuralPolicy(ShadowHandPolicy):
    """
    Minimal trainable policy for ShadowHand: flattened pixels -> MLP -> squashed Gaussian in (-1, 1).

    Samples in train mode and uses the mean in eval mode.

    Args:
        image_size: Transformed observation shape ``(C, H, W)``.
        hidden_size: MLP hidden width.
        num_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        c, h, w = image_size
        self.actor = ContinuousActor(
            feat_size=c * h * w,
            action_size=self.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        feat = obs["render"].flatten(start_dim=-3)
        return self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
