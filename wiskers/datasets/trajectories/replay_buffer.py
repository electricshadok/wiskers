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

    Internally backed by a collections.deque(maxlen=capacity). Adding a new
    episode to a full buffer automatically evicts the oldest one (FIFO, O(1)).

    The buffer solely stores episodes. Sampling is handled by pluggable
    `TrajectorySampler` implementations configured in Hydra via `_target_`.

    Args:
        capacity (int): Max number of episodes to store before evicting oldest.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.capacity = capacity
        self._episodes: deque[Episode] = deque(maxlen=capacity)

    def add(self, episode: Episode) -> None:
        """
        Store a finished episode in the buffer.

        If the buffer already holds `capacity` episodes, the oldest one is
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
        return f"ReplayBuffer(episodes={self.num_episodes}/{self.capacity}, steps={self.num_steps})"
