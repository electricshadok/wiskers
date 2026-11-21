"""
Random-shooting Model Predictive Control (MPC) planner.
"""

from typing import Any

import torch

from wiskers.envs.action_spec import BoundedActionSpec
from wiskers.planner.base import DynamicsModel, Planner


class RandomShootingPlanner(Planner):
    """
    Random-shooting MPC planner.

    Samples ``num_samples`` uniform candidate action trajectories of length
    ``horizon`` within ``action_spec``, scores them in imagination via
    ``model.imagine_returns``, and executes the first action of the
    highest-return trajectory.
    """

    def __init__(
        self,
        model: DynamicsModel,
        action_spec: BoundedActionSpec,
        horizon: int = 12,
        num_samples: int = 256,
    ) -> None:
        if not isinstance(action_spec, BoundedActionSpec):
            raise TypeError(f"RandomShootingPlanner requires BoundedActionSpec, got {type(action_spec).__name__}.")
        super().__init__(model=model, action_spec=action_spec, horizon=horizon, num_samples=num_samples)

    @property
    def action_spec(self) -> BoundedActionSpec:
        assert isinstance(self._action_spec, BoundedActionSpec)
        return self._action_spec

    def plan(self, state: Any) -> torch.Tensor:
        low = self.action_spec.low
        high = self.action_spec.high
        actions = (
            torch.rand(
                (self.num_samples, self.horizon, *self.action_spec.shape),
                dtype=self.action_spec.dtype,
                device=low.device,
            )
            * (high - low)
            + low
        )
        returns = self.model.imagine_returns(state, actions)
        best_idx = int(returns.argmax().item())
        return actions[best_idx, 0]
