"""
PyTorch IterableDataset backed by a ReplayBuffer.

Wraps the buffer in a thin Dataset so it can be used with a standard
DataLoader. Each iteration yields one Episode chunk of length num_steps.

The dataset streams samples indefinitely as an infinite generator.
Use the DataLoader's `batch_size` to control how many chunks form a batch,
and control training duration via Trainer `max_steps` or `limit_train_batches`.

    replay_buffer = ReplayBuffer(capacity_episodes=1000)
    sampler       = UniformChunkSampler(num_steps=50)
    transform     = EpisodeToTensor(image_size=(3, 64, 64))
    dataset       = TrajectoryDataset(replay_buffer=replay_buffer, sampler=sampler, transform=transform)
    loader        = DataLoader(dataset, batch_size=32, num_workers=0)

Note: num_workers must be 0 (or handled carefully) because the buffer
lives in the main process. Worker forks receive a snapshot of the buffer
at fork time and will not see new episodes added during training.
"""

from torch.utils.data import IterableDataset

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import TrajectorySampler
from wiskers.datasets.trajectories.transforms import EpisodeTransform


class TrajectoryDataset(IterableDataset):
    """
    Infinite-stream IterableDataset that draws random chunks from a ReplayBuffer.

    Each call to __iter__ yields individual Episode chunks as tensor dicts indefinitely.
    The DataLoader collates these into batches of shape (B, T, ...).

    Args:
        replay_buffer (ReplayBuffer): The replay buffer to sample from.
        sampler (TrajectorySampler): Sampling strategy to draw chunks.
        transform (EpisodeTransform): Transform converting an Episode chunk into tensors.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sampler: TrajectorySampler,
        transform: EpisodeTransform,
    ) -> None:
        super().__init__()
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.transform = transform

    def __iter__(self):
        while True:
            if self.replay_buffer.is_empty:
                raise RuntimeError(
                    "TrajectoryDataset: replay_buffer is empty. "
                    "Add episodes before starting training."
                )
            chunk = self.sampler.sample(self.replay_buffer, batch_size=1)[0]
            yield self.transform(chunk)
