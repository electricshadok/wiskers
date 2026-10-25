"""
ALE/Breakout-v5 live trajectory DataModule.
"""

from typing import Optional

from wiskers.datasets.gym.breakout_collector import collect_breakout_episodes
from wiskers.datasets.gym.breakout_policies import BreakoutPolicy
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
        replay_buffer: ReplayBuffer,
        policy: BreakoutPolicy,
        # Environment rollout configuration
        seed_episodes: int = 20,
        max_steps_per_episode: int = 500,
        image_size: Optional[tuple[int, int]] = (64, 64),
        # DataLoader / Training schedule
        batch_size: int = 32,
        steps_per_epoch: int = 500,
        # General settings
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            num_workers=0,
        )
        self.policy = policy
        self.seed_episodes = seed_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.image_size = tuple(image_size) if image_size is not None else None
        self.seed = seed
        self.verbose = verbose

    def setup(self, stage: Optional[str] = None) -> None:
        if self.replay_buffer.is_empty:
            collect_breakout_episodes(
                replay_buffer=self.replay_buffer,
                policy=self.policy,
                n_episodes=self.seed_episodes,
                max_steps=self.max_steps_per_episode,
                seed=self.seed,
                image_size=self.image_size,
                verbose=self.verbose,
            )
        super().setup(stage)
