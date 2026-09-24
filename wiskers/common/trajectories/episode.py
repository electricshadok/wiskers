"""
One complete environment rollout stored as numpy arrays.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """
    One complete environment rollout.

    Every field has T as its first dimension, where T is the number of
    environment steps taken during this episode (variable per episode).

    Fields:
        obs     (T, H, W, C) uint8     — raw pixel observations from the env.
        actions (T, action_dim) float32 — action taken at each step.
        rewards (T,) float32            — scalar reward received after each action.
        dones   (T,) bool               — True on the step the episode ended.
    """

    obs: np.ndarray      # (T, H, W, C) uint8
    actions: np.ndarray  # (T, action_dim) float32
    rewards: np.ndarray  # (T,) float32
    dones: np.ndarray    # (T,) bool

    def __len__(self) -> int:
        """Number of timesteps in this episode."""
        return len(self.obs)

    def slice(self, start: int, end: int) -> "Episode":
        """Return a contiguous sub-sequence of this episode covering steps [start, end)."""
        return Episode(
            obs=self.obs[start:end],
            actions=self.actions[start:end],
            rewards=self.rewards[start:end],
            dones=self.dones[start:end],
        )
