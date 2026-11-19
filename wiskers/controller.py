"""
Controller protocol for rollout action selection (policies or planners).
"""

from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import ActionSpec, DiscreteActionSpec


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


def validate_controller(controller: Controller, action_spec: ActionSpec) -> None:
    """Verify that ``controller.action_dim`` is compatible with ``action_spec``."""
    if isinstance(action_spec, DiscreteActionSpec):
        if controller.action_dim != 1:
            raise ValueError(
                f"Env has Discrete action spec, but controller has "
                f"action_dim={controller.action_dim} (expected 1)."
            )
    elif controller.action_dim != action_spec.action_dim:
        raise ValueError(
            f"Env has Bounded action spec with shape {action_spec.shape}, "
            f"but controller has action_dim={controller.action_dim} (expected {action_spec.action_dim})."
        )
