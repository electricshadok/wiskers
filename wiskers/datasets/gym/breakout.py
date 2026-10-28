"""
ALE/Breakout-v5 live trajectory DataModule.
"""

from typing import Mapping, Optional, Union

from wiskers.datasets.gym.breakout_collector import collect_breakout_episodes
from wiskers.datasets.gym.breakout_policies import BreakoutPolicy
from wiskers.datasets.trajectories.config import CollectionConfig, TrainingConfig
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler


class BreakoutDataModule(TrajectoryDataModule):
    """
    LightningDataModule for live ALE/Breakout-v5 environment rollouts.

    Wraps an in-memory ReplayBuffer and TrajectoryDataset. Automatically seeds
    the buffer with exploration rollouts during `setup()`, and allows
    online rollout callbacks to push new episodes dynamically during training.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
        policy: BreakoutPolicy,
        collection: Union[CollectionConfig, Mapping],
        training: Union[TrainingConfig, Mapping],
        image_size: Optional[tuple[int, int]] = (64, 64),
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            replay_buffer=replay_buffer,
            sampler=sampler,
            training=training,
        )
        self.policy = policy
        self.collection = CollectionConfig(**collection) if isinstance(collection, Mapping) else collection
        self.image_size = tuple(image_size) if image_size is not None else None
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> None:
        if self.replay_buffer.is_empty:
            collect_breakout_episodes(
                replay_buffer=self.replay_buffer,
                policy=self.policy,
                n_episodes=self.collection.seed_episodes,
                max_steps=self.collection.max_steps_per_episode,
                seed=self.seed,
                image_size=self.image_size,
                verbose=self.verbose,
            )
        super().setup(stage)
