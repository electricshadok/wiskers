"""
ALE/Breakout-v5 live trajectory DataModule.
"""

from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np

from wiskers.datasets.gym.breakout_collector import collect_breakout_episodes
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


class BreakoutDataModule(TrajectoryDataModule):
    """
    LightningDataModule for live ALE/Breakout-v5 environment rollouts.

    Wraps an in-memory ReplayBuffer and TrajectoryDataset. Automatically seeds
    the buffer with exploration rollouts during `setup()`, and allows
    online rollout callbacks to push new episodes dynamically during training.
    """

    def __init__(
        self,
        # Replay buffer configuration
        capacity: int = 1000,
        chunk_len: int = 50,
        # Environment rollout configuration
        seed_episodes: int = 20,
        max_steps_per_episode: int = 500,
        image_size: Optional[tuple[int, int]] = (64, 64),
        policy: Union[str, Callable[[np.ndarray, gym.Env, int], int]] = "heuristic",
        epsilon: float = 0.1,
        # DataLoader / Training schedule
        batch_size: int = 32,
        steps_per_epoch: int = 500,
        # General settings
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        buffer = ReplayBuffer(capacity=capacity, chunk_len=chunk_len)
        super().__init__(
            buffer=buffer,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            num_workers=0,
        )
        self.seed_episodes = seed_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.image_size = tuple(image_size) if image_size is not None else None
        self.policy = policy
        self.epsilon = epsilon
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> None:
        if self.buffer.is_empty:
            collect_breakout_episodes(
                buffer=self.buffer,
                n_episodes=self.seed_episodes,
                max_steps=self.max_steps_per_episode,
                seed=self.seed,
                image_size=self.image_size,
                policy=self.policy,
                epsilon=self.epsilon,
                verbose=self.verbose,
            )
        super().setup(stage)
