"""
Policies for InvertedPendulum environment.
"""

from typing import Sequence

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import BoundedActionSpec
from wiskers.nn.heads import ContinuousActor
from wiskers.policy.base import ContinuousPolicy


class InvertedPendulumPolicy(ContinuousPolicy):
    """Base policy for InvertedPendulum. Continuous 1D action space (cart force in [-3.0, 3.0] N)."""

    def __init__(self) -> None:
        super().__init__(
            BoundedActionSpec(
                low=torch.tensor([-3.0], dtype=torch.float32),
                high=torch.tensor([3.0], dtype=torch.float32),
            )
        )


class InvertedPendulumNeuralPolicy(InvertedPendulumPolicy):
    """
    Minimal trainable policy for InvertedPendulum: flattened pixels → MLP → squashed Gaussian scaled to (-3, 3).

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
        action = self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
        return action * 3.0
