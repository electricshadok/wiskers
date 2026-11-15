"""
Episode rollout container inheriting from TensorDict.
"""

from typing import Any, Optional, Sequence, Union

import torch
from tensordict import TensorDict


class Episode(TensorDict):
    """
    One complete environment rollout (or sliced sub-trajectory) of ``T`` transitions
    stored as a TensorDict with ``batch_size=[T]``.

    Structure:
        - "obs": Observation at step t [T, ...] (Tensor or TensorDict of modalities)
        - "actions": Action taken at step t [T, action_dim]
        - "next": Nested TensorDict containing transition outcomes:
            - "obs": Next observation at step t + 1 [T, ...]
            - "rewards": Scalar reward received at step t [T]
            - "done": Done flag at step t [T] (inferred as terminated | truncated)
            - "terminated": Terminal state reached at step t [T]
            - "truncated": Time-limit cutoff at step t [T]
    """

    REQUIRED_KEYS = (
        ("obs",),
        ("actions",),
        ("next", "obs"),
        ("next", "rewards"),
        ("next", "done"),
        ("next", "terminated"),
        ("next", "truncated"),
    )

    def __init__(
        self,
        obs: torch.Tensor | TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        *args: Any,
        batch_size: Optional[Union[torch.Size, Sequence[int]]] = None,
        **kwargs: Any,
    ) -> None:
        T = len(actions)
        source = {
            "obs": obs[:-1] if T > 0 else obs[:0],
            "actions": actions,
            "next": {
                "obs": obs[1:] if T > 0 else obs[:0],
                "rewards": rewards,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
        }
        super().__init__(source, *args, batch_size=[T] if batch_size is None else batch_size, **kwargs)
        self.validate()

    @classmethod
    def from_dict(
        cls,
        source: dict[str, Any],
        *,
        batch_size: Optional[Union[torch.Size, Sequence[int]]] = None,
        **kwargs: Any,
    ) -> "Episode":
        """Construct an Episode from an existing raw transition dictionary."""
        ep = cls.__new__(cls)
        TensorDict.__init__(ep, source, batch_size=batch_size, **kwargs)
        return ep

    def validate(self) -> None:
        """
        Validate that this Episode conforms to the transition layout:
            - Required keys exist: 'obs', 'actions', 'next' ('obs', 'rewards', 'done', 'terminated', 'truncated')
            - 'next' is a TensorDict
            - 'next/done' equals 'next/terminated | next/truncated'
        """
        for key in ("obs", "actions", "next"):
            if key not in self.keys():
                raise KeyError(f"Episode missing top-level key: '{key}'")

        nxt = self["next"]
        if not isinstance(nxt, TensorDict):
            raise ValueError(f"Episode 'next' must be a TensorDict, got {type(nxt)}")

        for key in ("obs", "rewards", "done", "terminated", "truncated"):
            if key not in nxt.keys():
                raise KeyError(f"Episode 'next' missing key: '{key}'")

        if not torch.equal(nxt["done"], nxt["terminated"] | nxt["truncated"]):
            raise ValueError("Episode 'next/done' must equal 'next/terminated | next/truncated'")

    def is_valid(self) -> bool:
        """Return True if this Episode satisfies all schema and consistency constraints."""
        try:
            self.validate()
            return True
        except (KeyError, ValueError):
            return False

    @property
    def obs(self) -> torch.Tensor | TensorDict:
        """Observations at step t [T, ...]."""
        return self["obs"]

    @property
    def actions(self) -> torch.Tensor:
        """Actions taken at step t [T, action_dim]."""
        return self["actions"]

    @property
    def rewards(self) -> torch.Tensor:
        """Scalar rewards received after step t [T]."""
        return self["next", "rewards"]

    @property
    def done(self) -> torch.Tensor:
        """Done flags at step t [T] stored in the TensorDict."""
        return self["next", "done"]

    @property
    def terminated(self) -> torch.Tensor:
        """Terminated flags at step t [T]."""
        return self["next", "terminated"]

    @property
    def truncated(self) -> torch.Tensor:
        """Truncated flags at step t [T]."""
        return self["next", "truncated"]

    @property
    def all_obs(self) -> torch.Tensor | TensorDict:
        """
        Full observation sequence [T + 1, ...], reconstructing the initial
        observation obs[0] concatenated with next.obs[0:T].
        """
        if len(self) == 0:
            return self["obs"]
        obs_curr = self["obs"]
        obs_next = self["next", "obs"]
        if isinstance(obs_curr, TensorDict):
            return torch.cat([obs_curr[:1], obs_next], dim=0)
        return torch.cat([obs_curr[:1], obs_next], dim=0)

    def slice(self, start: int, end: int) -> "Episode":
        """Return a contiguous sub-trajectory covering transitions [start, end)."""
        return self[start:end]
