"""
Controller protocol for rollout action selection (policies or planners).
"""

from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class Controller(Protocol):
    """
    Protocol for any component that produces actions from observations during rollouts
    (e.g., a ``Policy`` or a model-based ``Planner`` such as MPC / CEM).
    """

    action_dim: int

    def reset(self) -> None:
        """Reset any internal episodic state at the start of a rollout."""
        ...

    def __call__(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        """Select a 1D action tensor of shape ``(action_dim,)`` for the current observation."""
        ...
