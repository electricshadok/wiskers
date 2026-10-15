"""
CarRacing-v3 live trajectory DataModule.
"""

from typing import Optional

from wiskers.datasets.gym.car_racing_collector import (
    NoZoomCarRacing,
    collect_car_racing_episodes,
)
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


class CarRacingDataModule(TrajectoryDataModule):
    """
    LightningDataModule for live CarRacing-v3 environment rollouts.

    Wraps an in-memory ReplayBuffer and TrajectoryDataset. Automatically seeds
    the buffer with heuristic exploration rollouts during `setup()`, and allows
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
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> None:
        if self.buffer.is_empty:
            collect_car_racing_episodes(
                buffer=self.buffer,
                n_episodes=self.seed_episodes,
                max_steps=self.max_steps_per_episode,
                seed=self.seed,
                verbose=self.verbose,
            )
        super().setup(stage)


__all__ = ["CarRacingDataModule", "NoZoomCarRacing"]
