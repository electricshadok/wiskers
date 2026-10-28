"""
CarRacing-v3 live trajectory DataModule.
"""

from typing import Mapping, Optional, Union

from wiskers.datasets.gym.car_racing_collector import collect_car_racing_episodes
from wiskers.datasets.gym.car_racing_policies import CarRacingPolicy
from wiskers.datasets.trajectories.config import CollectionConfig, TrainingConfig
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler


class CarRacingDataModule(TrajectoryDataModule):
    """
    LightningDataModule for live CarRacing-v3 environment rollouts.

    Wraps an in-memory ReplayBuffer and TrajectoryDataset. Automatically seeds
    the buffer with heuristic exploration rollouts during `setup()`, and allows
    online rollout callbacks to push new episodes dynamically during training.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
        policy: CarRacingPolicy,
        collection: Union[CollectionConfig, Mapping],
        training: Union[TrainingConfig, Mapping],
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
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> None:
        if self.replay_buffer.is_empty:
            collect_car_racing_episodes(
                replay_buffer=self.replay_buffer,
                policy=self.policy,
                n_episodes=self.collection.seed_episodes,
                max_steps=self.collection.max_steps_per_episode,
                seed=self.seed,
                verbose=self.verbose,
            )
        super().setup(stage)
