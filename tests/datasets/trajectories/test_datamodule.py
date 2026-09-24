import numpy as np
import torch

from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


def _make_episode(T: int = 20, H: int = 8, W: int = 8, C: int = 3, action_dim: int = 4) -> Episode:
    """Build a dummy episode filled with random data."""
    return Episode(
        obs=np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8),
        actions=np.random.randn(T, action_dim).astype(np.float32),
        rewards=np.random.randn(T).astype(np.float32),
        dones=np.zeros(T, dtype=bool),
    )


def test_datamodule_train_dataloader():
    """
    Integration smoke-test: Episode → ReplayBuffer → TrajectoryDataModule → batch.

    Verifies that:
    - setup() creates the inner dataset without errors.
    - train_dataloader() yields a single batch with the expected keys and shapes.
    """
    CHUNK_LEN = 10
    BATCH_SIZE = 4
    STEPS_PER_EPOCH = BATCH_SIZE  # one batch worth of steps

    buffer = ReplayBuffer(capacity=10, chunk_len=CHUNK_LEN)
    buffer.add(_make_episode(T=30))

    dm = TrajectoryDataModule(buffer, batch_size=BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH)
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    # Keys
    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}

    # Shapes: (B, T, ...)
    assert batch["obs"].shape == (BATCH_SIZE, CHUNK_LEN, 8, 8, 3)
    assert batch["actions"].shape == (BATCH_SIZE, CHUNK_LEN, 4)
    assert batch["rewards"].shape == (BATCH_SIZE, CHUNK_LEN)
    assert batch["dones"].shape == (BATCH_SIZE, CHUNK_LEN)

    # dtypes
    assert batch["obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.float32
    assert batch["rewards"].dtype == torch.float32
    assert batch["dones"].dtype == torch.float32

    # Observations were uint8 [0,255] → normalised to [0,1]
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0
