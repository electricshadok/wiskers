"""
Buffers for trajectory rollouts and replay.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterator, Sequence, Union

from wiskers.data.episode import Episode


class Buffer(ABC):
    """
    Abstract base class for episode buffers.

    Subclasses manage storage (e.g. deque with fixed capacity for ReplayBuffer,
    or a list for RolloutBuffer). Episode indexing, iteration, and metrics
    are provided by the base class.
    """

    def __init__(
        self,
        episodes: Union[list[Episode], deque[Episode]],
    ) -> None:
        self._episodes = episodes

    def add(self, episode: Episode) -> None:
        """Store an episode in the buffer. Empty episodes are ignored."""
        if len(episode) > 0:
            self._episodes.append(episode)

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def reset(self) -> None:
        """Clear all stored episodes in the buffer."""
        self._episodes.clear()

    @property
    def is_empty(self) -> bool:
        """Return True if the buffer contains no episodes."""
        return len(self._episodes) == 0

    @property
    def num_episodes(self) -> int:
        """Number of episodes currently stored."""
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        """Total number of environment steps stored across all episodes."""
        return sum(len(ep) for ep in self._episodes)

    @property
    def episodes(self) -> Sequence[Episode]:
        """Sequence view of all currently stored episodes."""
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Episode, list[Episode]]:
        if isinstance(idx, slice):
            return list(self._episodes)[idx]
        return self._episodes[idx]


class ReplayBuffer(Buffer):
    """
    Fixed-capacity circular buffer of Episodes.

    Backed by a collections.deque(maxlen=capacity_episodes). Adding a new
    episode to a full buffer automatically evicts the oldest one (FIFO, O(1)).

    Args:
        capacity_episodes (int): Max number of complete episodes to store before evicting.
    """

    def __init__(self, capacity_episodes: int) -> None:
        super().__init__(deque(maxlen=capacity_episodes))
        self.capacity_episodes = capacity_episodes

    def __repr__(self) -> str:
        return f"ReplayBuffer(episodes={self.num_episodes}/{self.capacity_episodes}, total_steps={self.num_steps})"


class RolloutBuffer(Buffer):
    """
    Transient buffer that stores episodes collected during an active rollout.

    Stores episodes in a list. Typically cleared via reset() after processing
    each rollout batch or epoch.
    """

    def __init__(self) -> None:
        super().__init__([])

    def __repr__(self) -> str:
        return f"RolloutBuffer(episodes={self.num_episodes}, total_steps={self.num_steps})"
