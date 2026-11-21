"""
Base interfaces for model-based planners (e.g., Random Shooting, CEM, MPC).
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import ActionSpec


@runtime_checkable
class DynamicsModel(Protocol):
    """Minimal latent dynamics interface consumed by model-based planners."""

    def encode_step(
        self,
        obs: TensorDict,
        prev_action: torch.Tensor | None = None,
        state: Any = None,
    ) -> Any:
        """Update latent/recurrent state from the current observation and previous action."""
        ...

    def imagine_returns(self, state: Any, actions: torch.Tensor) -> torch.Tensor:
        """
        Roll out ``K`` candidate action sequences of shape ``(K, H, action_dim)`` in imagination
        from ``state`` and return predicted scalar returns of shape ``(K,)``.
        """
        ...


class Planner(ABC):
    """
    Abstract base class for model-based receding-horizon planners conforming to ``Controller``.

    Args:
        model: Latent dynamics model implementing ``DynamicsModel``.
        action_spec: Action specification produced by this planner.
        horizon: Planning horizon ``H`` (steps ahead).
        num_samples: Number of candidate action trajectories ``K`` evaluated per step.
    """

    def __init__(
        self,
        model: DynamicsModel,
        action_spec: ActionSpec,
        horizon: int = 12,
        num_samples: int = 256,
    ) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")

        self.model = model
        self._action_spec = action_spec
        self.horizon = horizon
        self.num_samples = num_samples

        self._state: Any = None
        self._prev_action: torch.Tensor | None = None

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the actions produced by this planner."""
        return self._action_spec

    def reset(self) -> None:
        """Reset episodic dynamics state at the start of a rollout."""
        self._state = None
        self._prev_action = None

    @abstractmethod
    def plan(self, state: Any) -> torch.Tensor:
        """Select an action tensor conforming to ``self.action_spec`` given the current latent ``state``."""
        ...

    @torch.inference_mode()
    def __call__(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        """Update dynamics state with ``obs`` and return the planned action for this step."""
        if step == 0:
            self.reset()
        self._state = self.model.encode_step(obs, self._prev_action, self._state)
        action = self.plan(self._state)
        self._prev_action = action.detach()
        return action
