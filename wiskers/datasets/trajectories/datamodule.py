"""
PyTorch Lightning DataModule for trajectory-based world-model training.

Owns the ReplayBuffer and the TrajectoryDataset. Does NOT interact with the
environment — episode collection is the caller's responsibility (typically
done via a Lightning Callback after each epoch).

Usage:

    replay_buffer = ReplayBuffer(capacity_episodes=1000)
    sampler = UniformChunkSampler(chunk_len=50)
    training = TrainingConfig(batch_size=32, steps_per_epoch=500)
    # ... populate replay_buffer with seed episodes ...

    dm = TrajectoryDataModule(replay_buffer=replay_buffer, sampler=sampler, training=training)
    trainer = Trainer(...)
    trainer.fit(model, datamodule=dm)

    # After each epoch, add new episodes via a Callback:
    #   dm.replay_buffer.add(new_episode)
"""

from typing import Mapping, Optional, Union

import lightning as L
from torch.utils.data import DataLoader

from wiskers.datasets.trajectories.config import TrainingConfig
from wiskers.datasets.trajectories.dataset import TrajectoryDataset
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler


class TrajectoryDataModule(L.LightningDataModule):
    """
    LightningDataModule that serves trajectory chunks from a ReplayBuffer.

    The DataModule holds a reference to the buffer so that a Lightning Callback
    can call dm.replay_buffer.add(episode) after each epoch to grow the dataset
    iteratively — without restarting the DataModule or the Trainer.

    Args:
        replay_buffer (ReplayBuffer): Shared buffer populated externally.
        sampler (TrajectorySampler): Sampling strategy to draw chunks.
                                     Configured in Hydra via `_target_`.
        training (TrainingConfig | dict): Batch size, steps_per_epoch, and workers.

    Note — val / test dataloaders:
        This DataModule intentionally omits val_dataloader() and test_dataloader().
        For world-model training, validation is measured by running the agent in
        the environment and recording cumulative reward — not by evaluating on
        held-out buffer data. Add a separate eval Callback for that.
        If you want to track world-model loss (reconstruction, KL) on a fixed
        set of episodes, pass a separate `val_buffer` and create a second
        TrajectoryDataset from it in val_dataloader().
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
        training: Union[TrainingConfig, Mapping],
    ) -> None:
        super().__init__()
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.training = TrainingConfig(**training) if isinstance(training, Mapping) else training

        self._dataset: Optional[TrajectoryDataset] = None

    @property
    def batch_size(self) -> int:
        """Batch size for DataLoader."""
        return self.training.batch_size

    @property
    def steps_per_epoch(self) -> int:
        """Number of batches to yield per epoch."""
        return self.training.steps_per_epoch

    @property
    def chunk_len(self) -> int:
        """Return the chunk length configured on the sampler."""
        return self.sampler.chunk_len

    def setup(self, stage: Optional[str] = None) -> None:
        self._dataset = TrajectoryDataset(
            replay_buffer=self.replay_buffer,
            sampler=self.sampler,
            steps_per_epoch=self.training.steps_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, "Call setup() before train_dataloader()"
        return DataLoader(
            self._dataset,
            batch_size=self.training.batch_size,
            num_workers=self.training.num_workers,
        )
