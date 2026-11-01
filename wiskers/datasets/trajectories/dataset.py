"""
PyTorch IterableDataset backed by a ReplayBuffer.

Wraps the buffer in a thin Dataset so it can be used with a standard
DataLoader. Each iteration yields one Episode chunk of length chunk_len.

The dataset streams samples indefinitely as an infinite generator.
Use the DataLoader's `batch_size` to control how many chunks form a batch,
and control training duration via Trainer `max_steps` or `limit_train_batches`.

    replay_buffer = ReplayBuffer(capacity_episodes=1000)
    sampler       = UniformChunkSampler(chunk_len=50)
    dataset       = TrajectoryDataset(replay_buffer=replay_buffer, sampler=sampler)
    loader        = DataLoader(dataset, batch_size=32, num_workers=0)

Note: num_workers must be 0 (or handled carefully) because the buffer
lives in the main process. Worker forks receive a snapshot of the buffer
at fork time and will not see new episodes added during training.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset

from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler


def _episode_to_tensors(episode: Episode) -> dict[str, torch.Tensor]:
    """
    Convert one Episode chunk (numpy arrays) into a dict of tensors on-the-fly.

    Post-processes observations into PyTorch channel-first format (T, C, H, W)
    and scales uint8 pixel values to float32 in [0.0, 1.0].
    """
    obs = torch.from_numpy(episode.obs)
    if obs.ndim == 4:
        # (T, H, W, C) -> (T, C, H, W)
        obs = obs.permute(0, 3, 1, 2)
    elif obs.ndim == 3:
        # (T, H, W) -> (T, 1, H, W)
        obs = obs.unsqueeze(1)

    return {
        "obs": obs.float() / 255.0,
        "actions": torch.from_numpy(episode.actions),
        "rewards": torch.from_numpy(episode.rewards),
        "dones": torch.from_numpy(episode.dones.astype(np.float32)),
    }


class TrajectoryDataset(IterableDataset):
    """
    Infinite-stream IterableDataset that draws random chunks from a ReplayBuffer.

    Each call to __iter__ yields individual Episode chunks as tensor dicts indefinitely.
    The DataLoader collates these into batches of shape (B, T, ...).

    Args:
        replay_buffer (ReplayBuffer): The replay buffer to sample from.
        sampler (TrajectorySampler): Sampling strategy to draw chunks.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
    ) -> None:
        super().__init__()
        self.replay_buffer = replay_buffer
        self.sampler = sampler

    def __iter__(self):
        while True:
            if self.replay_buffer.is_empty:
                raise RuntimeError(
                    "TrajectoryDataset: replay_buffer is empty. "
                    "Add episodes before starting training."
                )
            chunk = self.sampler.sample(self.replay_buffer, batch_size=1)[0]
            yield _episode_to_tensors(chunk)
