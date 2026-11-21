"""
Policies for AdroitHand dexterous manipulation environments.
"""

from typing import Sequence

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import BoundedActionSpec
from wiskers.nn.heads import ContinuousActor
from wiskers.policy.base import ContinuousPolicy


class AdroitHandPolicy(ContinuousPolicy):
    """Base policy for AdroitHand. Continuous action space (24..30 DoF depending on task)."""

    def __init__(self, action_dim: int = 28) -> None:
        super().__init__(
            BoundedActionSpec(
                low=torch.full((action_dim,), -1.0, dtype=torch.float32),
                high=torch.full((action_dim,), 1.0, dtype=torch.float32),
            )
        )


class AdroitHandRandomPolicy(AdroitHandPolicy):
    """Uniform random exploration for AdroitHand."""

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        return self.action_spec.sample()


class AdroitHandHeuristicPolicy(AdroitHandPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for AdroitHand control.

    Generates coherent reaching and finger-articulation trajectories rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
        action_dim: Action dimensionality.
    """

    def __init__(
        self,
        momentum: float = 0.85,
        noise_scale: float = 0.25,
        action_dim: int = 28,
    ) -> None:
        super().__init__(action_dim=action_dim)
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)

    def reset(self) -> None:
        self.current_action = torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        noise = torch.randn(self.action_spec.shape, dtype=self.action_spec.dtype) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = torch.clamp(action, min=self.action_spec.low, max=self.action_spec.high)
        return self.current_action.clone()


class AdroitHandNeuralPolicy(AdroitHandPolicy):
    """
    Minimal trainable policy for AdroitHand: flattened pixels → MLP → squashed Gaussian in (-1, 1).

    Samples in train mode and uses the mean in eval mode.

    Args:
        image_size: Transformed observation shape ``(C, H, W)``.
        action_dim: Action dimensionality of the target task (e.g. 28 for ``door``).
        hidden_size: MLP hidden width.
        num_layers: Number of MLP hidden layers.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        action_dim: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__(action_dim=action_dim)
        c, h, w = image_size
        self.actor = ContinuousActor(
            feat_size=c * h * w,
            action_size=self.action_spec.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        feat = obs["render"].flatten(start_dim=-3)
        return self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
