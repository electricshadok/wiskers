"""
Fixed-capacity circular buffer of Episodes.

Episodes are stored as complete rollouts and sampled as fixed-length
contiguous chunks for sequence model training (RSSM, Dreamer, DIAMOND, ...).
"""

import random
from collections import deque

from wiskers.common.trajectories.episode import Episode


class ReplayBuffer:
    """
    Fixed-capacity circular buffer of Episodes.

    Internally backed by a collections.deque(maxlen=capacity). Adding a new
    episode to a full buffer automatically evicts the oldest one (FIFO, O(1)).

    When sampling, episodes are not picked uniformly — they are weighted by how
    many valid chunk start positions they contain:

        weight(ep) = len(ep) - chunk_len + 1

    This ensures every individual (episode, start_index) pair has equal
    probability, so long episodes are not under-sampled relative to short ones.

    Args:
        capacity  (int): Max number of episodes to store before evicting oldest.
        chunk_len (int): Length of the fixed-size sequence returned by sample().
    """

    def __init__(self, capacity: int, chunk_len: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")

        self.capacity = capacity
        self.chunk_len = chunk_len

        # Both deques share the same maxlen so eviction stays in sync:
        # when an episode is dropped from _episodes, its weight is dropped too.
        self._episodes: deque[Episode] = deque(maxlen=capacity)
        self._weights: deque[int] = deque(maxlen=capacity)

    def add(self, episode: Episode) -> None:
        """
        Store a finished episode in the buffer.

        If the buffer already holds `capacity` episodes, the oldest one is
        evicted automatically before the new one is inserted.

        Episodes shorter than chunk_len are silently ignored — they cannot
        produce a valid training chunk so keeping them would waste memory
        and distort sampling weights.
        """
        if len(episode) < self.chunk_len:
            return  # episode too short to sample from — skip silently
        self._episodes.append(episode)
        self._weights.append(len(episode) - self.chunk_len + 1)

    def sample(self, batch_size: int) -> list[Episode]:
        """
        Return batch_size random fixed-length chunks from the buffer.

        Each chunk is sampled in two steps:
            1. Pick an episode with probability proportional to its number of
               valid start positions (longer episodes → more likely to be picked).
            2. Pick a uniformly random start index within that episode so the
               returned chunk of length chunk_len fits entirely inside it.

        This two-level sampling guarantees that every (episode, start) pair in
        the buffer has exactly equal probability — no transition is over- or
        under-represented regardless of episode length.

        Raises RuntimeError if the buffer has no episodes yet.
        """
        if not self._episodes:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")

        chunks = []
        for _ in range(batch_size):
            episode = random.choices(self._episodes, weights=self._weights, k=1)[0]
            max_start = len(episode) - self.chunk_len
            start = random.randint(0, max_start)
            chunks.append(episode.slice(start, start + self.chunk_len))

        return chunks

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
        return (
            f"ReplayBuffer("
            f"episodes={self.num_episodes}/{self.capacity}, "
            f"steps={self.num_steps}, "
            f"chunk_len={self.chunk_len})"
        )
