"""
Replay buffer for world-model / RL training.

World models like Dreamer and PlaNet are trained on sequences of experience
collected by running an agent in an environment. This module provides:

    Episode        — dataclass that holds one complete game run (obs, actions,
                     rewards, dones) as numpy arrays of shape (T, ...).

    ReplayBuffer   — fixed-capacity circular buffer of Episodes.  When full,
                     the oldest episode is silently evicted to make room for
                     the new one.  Sampling returns fixed-length contiguous
                     chunks so sequence models (RSSM, GRU, Transformer) can
                     unroll over exactly chunk_len steps.

Quick start:

    buffer = ReplayBuffer(capacity=1000, chunk_len=50)
    buffer.add(episode)               # store a finished episode
    chunks = buffer.sample(32)        # list of 32 Episode chunks of length 50
"""

import random
from collections import deque
from dataclasses import dataclass

import numpy as np

@dataclass
class Episode:
    """
    One complete environment rollout.

    Every field has T as its first dimension, where T is the number of
    environment steps taken during this episode (variable per episode).

    Fields:
        obs     (T, H, W, C) uint8    — raw pixel observations from the env.
        actions (T, action_dim) float32 — action taken at each step.
        rewards (T,) float32           — scalar reward received after each action.
        dones   (T,) bool              — True on the step the episode ended.
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

class ReplayBuffer:
    """
    Fixed-capacity circular buffer of Episodes.

    Internally backed by a collections.deque(maxlen=capacity).  Adding a new
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
            start = random.randint(0, max_start)              # random start
            chunks.append(episode.slice(start, start + self.chunk_len))

        return chunks

    # Properties

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

    # Dunder helpers

    def __len__(self) -> int:
        """Number of episodes in the buffer (same as ``num_episodes``)."""
        return len(self._episodes)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer("
            f"episodes={self.num_episodes}/{self.capacity}, "
            f"steps={self.num_steps}, "
            f"chunk_len={self.chunk_len})"
        )
