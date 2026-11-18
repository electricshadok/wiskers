"""
Episode rollout container inheriting from TensorDict.
"""

from typing import Any

import torch
from tensordict import TensorDict


def _check_obs_batch_size(obs: Any, T: int) -> None:
    """Validate that observation TensorDict has explicit batch_size=[T + 1]."""
    if not isinstance(obs, TensorDict):
        raise TypeError(f"obs must be a TensorDict, got {type(obs).__name__}")
    expected_batch_size = torch.Size([T + 1])
    if obs.batch_size != expected_batch_size and not (T == 0 and obs.batch_size in (torch.Size([0]), torch.Size([1]))):
        raise ValueError(
            f"obs TensorDict must have explicit batch_size=[{T + 1}], got batch_size={list(obs.batch_size)}"
        )


def _check_action_tensor(action: Any, T: int) -> None:
    """Validate that action tensor is 2D with shape (T, action_dim)."""
    if not isinstance(action, torch.Tensor) or action.ndim < 2 or len(action) != T:
        shape_info = getattr(action, "shape", type(action))
        raise ValueError(f"action must be a torch.Tensor with shape ({T}, action_dim), got {shape_info}")


def _check_signal_tensor(name: str, tensor: Any, T: int) -> None:
    """Validate that transition signal tensor (reward, terminated, truncated) has shape (T, 1)."""
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (T, 1):
        shape_info = getattr(tensor, "shape", type(tensor))
        raise ValueError(f"{name} must be a torch.Tensor with shape ({T}, 1), got {shape_info}")


class Episode(TensorDict):
    """
    One complete environment rollout (or sliced sub-trajectory) of ``T`` transitions
    stored as a TensorDict with ``batch_size=[T]`` following TorchRL transition conventions.

    Batch Dimension Hierarchy:
        - Single transition:   batch_size=[]     | action: [action_dim], reward: [1], done: [1]
        - Trajectory sequence: batch_size=[T]    | action: [T, action_dim], reward: [T, 1], done: [T, 1]
        - Batched trajectory:  batch_size=[B, T] | action: [B, T, action_dim], reward: [B, T, 1]

    Structure:
        - "obs": TensorDict of observations at step t [T, ...]
        - "action": Action taken at step t [T, action_dim]
        - "next": Nested TensorDict containing transition outcomes:
            - "obs": Next observation TensorDict at step t + 1 [T, ...]
            - "reward": Scalar reward received at step t [T, 1] (float32)
            - "done": Done flag at step t [T, 1] (bool, inferred as terminated | truncated)
            - "terminated": Terminal state reached at step t [T, 1] (bool)
            - "truncated": Time-limit cutoff at step t [T, 1] (bool)
    """

    def __init__(
        self,
        obs: TensorDict,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """
        Construct an Episode of T transitions from a trajectory rollout.

        An episode of T transitions consists of:
            - T + 1 observations: o_0, o_1, ..., o_T
            - T actions: a_0, ..., a_{T-1}
            - T transition outcomes: r_0, done_0, ...

        The observations are time-sliced into:
            - obs: current-step observations o_0, ..., o_{T-1}  [T, ...]
            - next/obs: next-step observations o_1, ..., o_T    [T, ...]

        Args:
            obs (TensorDict): Full observation sequence of T + 1 steps across arbitrary
                modalities with explicit batch_size=[T + 1].
            action (torch.Tensor): Actions taken at each transition [T, action_dim].
            reward (torch.Tensor): Scalar rewards received at each transition [T, 1].
            terminated (torch.Tensor): Terminal state flags at each transition [T, 1].
            truncated (torch.Tensor): Time-limit truncation flags at each transition [T, 1].
        """
        T = len(action)
        _check_obs_batch_size(obs, T)
        _check_action_tensor(action, T)
        _check_signal_tensor("reward", reward, T)
        _check_signal_tensor("terminated", terminated, T)
        _check_signal_tensor("truncated", truncated, T)

        done = terminated | truncated

        nxt = TensorDict(
            {
                "obs": obs[1:],
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=[T],
        )

        source = {
            "obs": obs[:-1],
            "action": action,
            "next": nxt,
        }
        super().__init__(source, batch_size=[T])
        self.validate()

    def validate(self) -> None:
        """
        Validate that this Episode conforms to TorchRL transition layout:
            - Required top-level keys exist: 'obs', 'action', 'next'
            - 'obs' is a TensorDict matching parent batch_size
            - 'action' is a torch.Tensor
            - 'next' is a TensorDict matching parent batch_size
            - 'next/obs' is a TensorDict matching 'obs' keys and batch_size
            - 'reward' is a floating-point tensor with trailing singleton dimension (*, 1)
            - 'done', 'terminated', 'truncated' are boolean tensors with trailing singleton dimension (*, 1)
            - 'next/done' equals 'next/terminated | next/truncated'
        """
        # 1. Required top-level keys ('obs', 'action', 'next')
        for key in ("obs", "action", "next"):
            if key not in self.keys():
                raise KeyError(f"Episode missing top-level key: '{key}'")

        # 2. Current observations ('obs'): must be a TensorDict with matching batch_size
        obs = self["obs"]
        if not isinstance(obs, TensorDict):
            raise ValueError(f"Episode 'obs' must be a TensorDict, got {type(obs)}")
        if obs.batch_size != self.batch_size:
            raise ValueError(f"Episode 'obs' batch_size {list(obs.batch_size)} must match {list(self.batch_size)}")

        # 3. Actions ('action'): must be a torch.Tensor
        if not isinstance(self["action"], torch.Tensor):
            raise ValueError(f"Episode 'action' must be a torch.Tensor, got {type(self['action'])}")

        # 4. Transition outcomes ('next'): must be a TensorDict with matching batch_size and required keys
        nxt = self["next"]
        if not isinstance(nxt, TensorDict):
            raise ValueError(f"Episode 'next' must be a TensorDict, got {type(nxt)}")
        if nxt.batch_size != self.batch_size:
            raise ValueError(f"Episode 'next' batch_size {list(nxt.batch_size)} must match {list(self.batch_size)}")

        for key in ("obs", "reward", "done", "terminated", "truncated"):
            if key not in nxt.keys():
                raise KeyError(f"Episode 'next' missing key: '{key}'")

        # 5. Next observations ('next/obs'): must be a TensorDict matching 'obs' batch_size and keys
        nxt_obs = nxt["obs"]
        if not isinstance(nxt_obs, TensorDict):
            raise ValueError(f"Episode 'next/obs' must be a TensorDict, got {type(nxt_obs)}")
        if nxt_obs.batch_size != self.batch_size:
            raise ValueError(
                f"Episode 'next/obs' batch_size {list(nxt_obs.batch_size)} must match {list(self.batch_size)}"
            )

        if set(obs.keys()) != set(nxt_obs.keys()):
            raise KeyError("Episode 'obs' and 'next/obs' must have matching keys")

        for key, val in obs.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(f"Episode 'obs/{key}' must be a torch.Tensor (flat obs), got {type(val)}")

        # 6. Transition signals ('reward', 'done', 'terminated', 'truncated'): trailing (*, 1) shape and dtypes
        for flag_key in ("reward", "done", "terminated", "truncated"):
            t = nxt[flag_key]
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"Episode 'next/{flag_key}' must be a torch.Tensor, got {type(t)}")
            if t.ndim < 1 or t.shape[-1] != 1:
                raise ValueError(
                    f"Episode 'next/{flag_key}' must have trailing singleton dimension (*, 1), got shape {t.shape}"
                )

        if not nxt["reward"].is_floating_point():
            raise TypeError(f"Episode 'next/reward' must be a floating-point tensor, got {nxt['reward'].dtype}")

        for bool_key in ("done", "terminated", "truncated"):
            if nxt[bool_key].dtype != torch.bool:
                raise TypeError(f"Episode 'next/{bool_key}' must be a boolean tensor, got {nxt[bool_key].dtype}")

        # 7. Logical consistency: 'next/done' must equal 'next/terminated | next/truncated'
        if not torch.equal(nxt["done"], nxt["terminated"] | nxt["truncated"]):
            raise ValueError("Episode 'next/done' must equal 'next/terminated | next/truncated'")

    def is_valid(self) -> bool:
        """Return True if this Episode satisfies all schema and consistency constraints."""
        try:
            Episode.validate(self)
            return True
        except (KeyError, ValueError, TypeError):
            return False

    @property
    def obs(self) -> TensorDict:
        """Observations TensorDict at step t [T, ...]."""
        return self["obs"]

    @property
    def action(self) -> torch.Tensor:
        """Actions taken at step t [T, action_dim]."""
        return self["action"]

    @property
    def reward(self) -> torch.Tensor:
        """Scalar rewards received after step t [T, 1]."""
        return self["next", "reward"]

    @property
    def done(self) -> torch.Tensor:
        """Done flags at step t [T, 1] stored in the TensorDict."""
        return self["next", "done"]

    @property
    def terminated(self) -> torch.Tensor:
        """Terminated flags at step t [T, 1]."""
        return self["next", "terminated"]

    @property
    def truncated(self) -> torch.Tensor:
        """Truncated flags at step t [T, 1]."""
        return self["next", "truncated"]

    def slice(self, start: int, end: int) -> "Episode":
        """Return a contiguous sub-trajectory covering transitions [start, end)."""
        return self[start:end]
