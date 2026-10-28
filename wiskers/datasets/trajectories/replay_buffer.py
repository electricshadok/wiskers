"""
Fixed-capacity circular buffer of Episodes.

Stores complete rollouts. Sampling is delegated to pluggable TrajectorySamplers.
"""

from collections import deque
from typing import Iterator, Sequence, Union

from wiskers.datasets.trajectories.episode import Episode


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of Episodes.

    Internally backed by a collections.deque(maxlen=capacity_episodes). Adding a new
    episode to a full buffer automatically evicts the oldest one (FIFO, O(1)).

    Capacity is explicitly measured in **episodes** (complete rollout trajectories).

    The buffer solely stores episodes. Sampling is handled by pluggable
    `TrajectorySampler` implementations configured in Hydra via `_target_`.

    Args:
        capacity_episodes (int): Max number of complete episodes (rollouts) to
                                 store before evicting the oldest (FIFO).
    """

    def __init__(self, capacity_episodes: int) -> None:
        if capacity_episodes < 1:
            raise ValueError(f"capacity_episodes must be >= 1, got {capacity_episodes}")

        self.capacity_episodes = capacity_episodes
        self._episodes: deque[Episode] = deque(maxlen=capacity_episodes)

    def add(self, episode: Episode) -> None:
        """
        Store a finished episode in the buffer.

        If the buffer already holds `capacity_episodes` episodes, the oldest one is
        evicted automatically before the new one is inserted. Empty episodes
        are ignored.
        """
        if len(episode) == 0:
            return
        self._episodes.append(episode)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Episode, list[Episode]]:
        if isinstance(idx, slice):
            return list(self._episodes)[idx]
        return self._episodes[idx]

    @property
    def episodes(self) -> Sequence[Episode]:
        """Sequence view of all currently stored episodes."""
        return list(self._episodes)

    @property
    def num_episodes(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        """Total number of environment steps stored across all episodes."""
        return sum(len(ep) for ep in self._episodes)

    @property
    def is_empty(self) -> bool:
        """True if no episodes have been added yet."""
        return len(self._episodes) == 0

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"ReplayBuffer(episodes={self.num_episodes}/{self.capacity_episodes}, total_steps={self.num_steps})"
