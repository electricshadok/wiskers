"""
Controller protocol for rollout action selection (policies or planners).
"""

from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import ActionSpec


@runtime_checkable
class Controller(Protocol):
    """
    Protocol for any component that produces actions from observations during rollouts
    (e.g., a ``Policy`` or a model-based ``Planner`` such as MPC / CEM).
    """

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the actions produced by this controller."""
        ...

    def reset(self) -> None:
        """Reset any internal episodic state at the start of a rollout."""
        ...

    def __call__(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        """Select a 1D action tensor conforming to ``self.action_spec`` for the current observation."""
        ...


def validate_controller(controller: Controller, action_spec: ActionSpec) -> None:
    """Verify that ``controller.action_spec`` matches the environment's ``action_spec``."""
    if controller.action_spec != action_spec:
        raise ValueError(
            f"Controller action_spec ({controller.action_spec}) does not match "
            f"environment action_spec ({action_spec})."
        )
