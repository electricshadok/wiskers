"""
One complete environment rollout stored as numpy arrays.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """
    One complete environment rollout (or sliced sub-trajectory) of ``T`` transitions:

        o_0 --(a_0, r_0, d_0)--> o_1 --(a_1, r_1, d_1)--> ... --(a_{T-1}, r_{T-1}, d_{T-1})--> o_T

    Because each step ``t`` transitions from ``o_t`` to ``o_{t+1}``, ``obs`` has
    leading dimension ``T + 1`` while ``actions``, ``rewards``, and ``dones``
    have leading dimension ``T``.

    Fields:
        obs     (T + 1, H, W, C) uint8  — pixel observations [o_0, ..., o_T].
        actions (T, action_dim) float32 — action taken at each step [a_0, ..., a_{T-1}].
        rewards (T,) float32            — scalar reward received after each action.
        dones   (T,) bool               — True on the step the episode ended.
    """

    obs: np.ndarray      # (T + 1, H, W, C) uint8
    actions: np.ndarray  # (T, action_dim) float32
    rewards: np.ndarray  # (T,) float32
    dones: np.ndarray    # (T,) bool

    def __post_init__(self) -> None:
        T = len(self.actions)
        if len(self.rewards) != T or len(self.dones) != T:
            raise ValueError(
                f"Episode actions ({T}), rewards ({len(self.rewards)}), and "
                f"dones ({len(self.dones)}) must have the same length T."
            )
        if T == 0:
            if len(self.obs) not in (0, 1):
                raise ValueError(
                    f"Empty Episode (T=0) must have 0 or 1 observations, got {len(self.obs)}."
                )
        elif len(self.obs) != T + 1:
            raise ValueError(
                f"Episode with T={T} transitions must have T + 1 = {T + 1} observations, "
                f"got {len(self.obs)}."
            )

    def __len__(self) -> int:
        """Number of environment transitions (steps) T in this episode."""
        return len(self.actions)

    def slice(self, start: int, end: int) -> "Episode":
        """
        Return a contiguous sub-sequence covering transitions ``[start, end)``.

        For ``H = end - start`` transitions, the returned Episode contains:
            - ``obs``: ``self.obs[start : end + 1]`` of length ``H + 1``
            - ``actions``: ``self.actions[start : end]`` of length ``H``
            - ``rewards``: ``self.rewards[start : end]`` of length ``H``
            - ``dones``: ``self.dones[start : end]`` of length ``H``
        """
        if start >= end:
            return Episode(
                obs=self.obs[start:start],
                actions=self.actions[start:start],
                rewards=self.rewards[start:start],
                dones=self.dones[start:start],
            )
        return Episode(
            obs=self.obs[start : end + 1],
            actions=self.actions[start:end],
            rewards=self.rewards[start:end],
            dones=self.dones[start:end],
        )
