"""
Base policy interfaces for discrete and continuous environment rollouts.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict

from wiskers.envs.action_spec import ActionSpec, BoundedActionSpec, DiscreteActionSpec


class Policy(nn.Module):
    """
    Base class for all rollout policies (random, heuristic, or learned).

    Args:
        action_spec: Specification of the action tensor produced by this policy.
    """

    def __init__(self, action_spec: ActionSpec) -> None:
        super().__init__()
        self._action_spec = action_spec
        self.eval()

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the actions produced by this policy."""
        return self._action_spec

    def reset(self) -> None:
        """Reset any internal recurrent or episodic state at the start of an episode."""
        self.eval()

    def forward(
        self,
        obs: TensorDict,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute the action tensor for a single environment step.

        Args:
            obs (TensorDict): Observation TensorDict for the current step
                (e.g., ``obs["render"]`` of shape ``(C, H, W)`` and/or state tensors).
            step (int): Current step index within the episode.

        Returns:
            torch.Tensor: 1D action tensor conforming to ``self.action_spec``.
                - Discrete: shape ``(1,)`` with integer dtype (e.g., ``torch.tensor([2])``).
                - Continuous: shape ``action_spec.shape`` with float/numeric dtype.
        """
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Policy for discrete action spaces returning a 1D integer ``torch.Tensor`` of shape ``(1,)``."""

    def __init__(self, action_spec: DiscreteActionSpec) -> None:
        if not isinstance(action_spec, DiscreteActionSpec):
            raise TypeError(f"DiscretePolicy requires DiscreteActionSpec, got {type(action_spec).__name__}.")
        super().__init__(action_spec)

    @property
    def action_spec(self) -> DiscreteActionSpec:
        assert isinstance(self._action_spec, DiscreteActionSpec)
        return self._action_spec


class ContinuousPolicy(Policy):
    """Policy for continuous/bounded action spaces returning a ``torch.Tensor`` conforming to ``self.action_spec``."""

    def __init__(self, action_spec: BoundedActionSpec) -> None:
        if not isinstance(action_spec, BoundedActionSpec):
            raise TypeError(f"ContinuousPolicy requires BoundedActionSpec, got {type(action_spec).__name__}.")
        super().__init__(action_spec)

    @property
    def action_spec(self) -> BoundedActionSpec:
        assert isinstance(self._action_spec, BoundedActionSpec)
        return self._action_spec


class RandomPolicy(Policy):
    """Uniform random exploration across an ``ActionSpec``."""

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        """
        Sample an action uniformly at random.

        Returns:
            torch.Tensor: Action conforming to ``self.action_spec``.
        """
        return self.action_spec.sample()
