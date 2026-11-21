"""
Policies for FrankaKitchen environment.
"""

from typing import Sequence

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import BoundedActionSpec
from wiskers.nn.heads import ContinuousActor
from wiskers.policy.base import ContinuousPolicy


class FrankaKitchenPolicy(ContinuousPolicy):
    """Base policy for FrankaKitchen. Continuous 9D action space (7 arm joints + 2 gripper joints)."""

    def __init__(self) -> None:
        super().__init__(
            BoundedActionSpec(
                low=torch.full((9,), -1.0, dtype=torch.float32),
                high=torch.full((9,), 1.0, dtype=torch.float32),
            )
        )


class FrankaKitchenRandomPolicy(FrankaKitchenPolicy):
    """Uniform random exploration for FrankaKitchen."""

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        return self.action_spec.sample()


class FrankaKitchenHeuristicPolicy(FrankaKitchenPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for continuous 9D control.

    Generates coherent, smooth reaching and grasping movements rather than high-frequency jitter.

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
        self.current_action: torch.Tensor = torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)

    def reset(self) -> None:
        self.current_action = torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        noise = torch.randn(self.action_spec.shape, dtype=self.action_spec.dtype) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = torch.clamp(action, min=self.action_spec.low, max=self.action_spec.high)
        return self.current_action.clone()


class FrankaKitchenNeuralPolicy(FrankaKitchenPolicy):
    """
    Minimal trainable policy for FrankaKitchen: flattened pixels → MLP → squashed Gaussian in (-1, 1).

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
            action_size=self.action_spec.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        feat = obs["render"].flatten(start_dim=-3)
        return self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
