"""
Random-shooting Model Predictive Control (MPC) planner.
"""

from typing import Any

import torch

from wiskers.planner.base import Planner


class RandomShootingPlanner(Planner):
    """
    Random-shooting MPC planner.

    Samples ``num_samples`` uniform candidate action trajectories of length
    ``horizon``, scores them in imagination via ``model.imagine_returns``, and
    executes the first action of the highest-return trajectory.
    """

    def plan(self, state: Any) -> torch.Tensor:
        actions = (
            torch.rand((self.num_samples, self.horizon, self.action_dim), dtype=torch.float32)
            * (self.high - self.low)
            + self.low
        )
        returns = self.model.imagine_returns(state, actions)
        best_idx = int(returns.argmax().item())
        return actions[best_idx, 0]
