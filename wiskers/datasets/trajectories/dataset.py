"""
PyTorch IterableDataset backed by a ReplayBuffer.

Wraps the buffer in a thin Dataset so it can be used with a standard
DataLoader. Each iteration yields one Episode chunk of length chunk_len.

The dataset has no fixed epoch size — it streams samples indefinitely.
Use the DataLoader's `batch_size` to control how many chunks form a batch,
and stop iteration externally (e.g. via trainer max_steps or a step limit).

    buffer  = ReplayBuffer(capacity=1000, chunk_len=50)
    dataset = TrajectoryDataset(buffer)
    loader  = DataLoader(dataset, batch_size=32, num_workers=0)

Note: num_workers must be 0 (or handled carefully) because the buffer
lives in the main process. Worker forks receive a snapshot of the buffer
at fork time and will not see new episodes added during training.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset

from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


def _episode_to_tensors(episode: Episode) -> dict[str, torch.Tensor]:
    """Convert one Episode chunk (numpy arrays) into a dict of tensors."""
    return {
        "obs":     torch.from_numpy(episode.obs.astype(np.float32) / 255.0),
        "actions": torch.from_numpy(episode.actions),
        "rewards": torch.from_numpy(episode.rewards),
        "dones":   torch.from_numpy(episode.dones.astype(np.float32)),
    }


class TrajectoryDataset(IterableDataset):
    """
    Infinite-stream IterableDataset that draws random chunks from a ReplayBuffer.

    Each call to __iter__ yields individual Episode chunks as tensor dicts.
    The DataLoader collates these into batches of shape (B, T, ...).

    Args:
        buffer      (ReplayBuffer): The buffer to sample from.
        steps_per_epoch (int): Number of chunks to yield per epoch (i.e. per
                               DataLoader pass). Controls how often Lightning
                               runs validation and logs metrics.
    """

    def __init__(self, buffer: ReplayBuffer, steps_per_epoch: int = 1000) -> None:
        super().__init__()
        self.buffer = buffer
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            if self.buffer.is_empty:
                raise RuntimeError(
                    "TrajectoryDataset: buffer is empty. "
                    "Add episodes before starting training."
                )
            chunk = self.buffer.sample(batch_size=1)[0]
            yield _episode_to_tensors(chunk)
