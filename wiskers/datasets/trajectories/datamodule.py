"""
PyTorch Lightning DataModule for trajectory-based world-model training.

Owns the ReplayBuffer and the TrajectoryDataset. Does NOT interact with the
environment — episode collection is the caller's responsibility (typically
done via a Lightning Callback after each epoch).

Usage:

    replay_buffer = ReplayBuffer(capacity_episodes=1000)
    sampler = UniformChunkSampler(num_steps=50)
    # ... populate replay_buffer with seed episodes ...

    dm = TrajectoryDataModule(replay_buffer=replay_buffer, sampler=sampler, batch_size=32)
    trainer = Trainer(...)
    trainer.fit(model, datamodule=dm)

    # After each epoch, add new episodes via a Callback:
    #   dm.replay_buffer.add(new_episode)
"""

from typing import Any, Optional

import lightning as L
from torch.utils.data import DataLoader

from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.dataset import TrajectoryDataset
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler


class TrajectoryDataModule(L.LightningDataModule):
    """
    LightningDataModule that serves trajectory chunks from a ReplayBuffer.

    Optionally accepts a Collector to seed the ReplayBuffer during `setup()`
    if the buffer is empty, and allows online rollout callbacks to push new
    episodes dynamically during training.

    Args:
        replay_buffer (ReplayBuffer): Shared buffer populated externally or via collector.
        sampler (TrajectorySampler): Sampling strategy to draw chunks.
        collector (Optional[Collector]): Collector to seed the buffer during setup().
        batch_size (int): Number of chunks per training batch (default: 32).
        num_workers (int): DataLoader workers (default: 0).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
        collector: Optional[Collector] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        if collector is not None:
            if sampler.num_steps > collector.max_steps:
                raise ValueError(
                    f"sampler.num_steps ({sampler.num_steps}) cannot be greater than "
                    f"collector.max_steps ({collector.max_steps})."
                )
            if collector.min_steps < sampler.num_steps:
                raise ValueError(
                    f"collector.min_steps ({collector.min_steps}) cannot be less than "
                    f"sampler.num_steps ({sampler.num_steps}). Episodes shorter than "
                    f"sampler.num_steps cannot be sampled."
                )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.collector = collector
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._dataset: Optional[TrajectoryDataset] = None

    @property
    def num_steps(self) -> int:
        """Return the sequence length configured on the sampler."""
        return self.sampler.num_steps

    @property
    def policy(self) -> Any:
        return self.collector.policy if self.collector is not None else None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.collector is not None and self.replay_buffer.is_empty:
            self.collector.collect(self.replay_buffer)

        self._dataset = TrajectoryDataset(
            replay_buffer=self.replay_buffer,
            sampler=self.sampler,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "Call setup() before train_dataloader()"
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
