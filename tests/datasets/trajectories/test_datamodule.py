import numpy as np
import pytest
import torch

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.sampler import UniformChunkSampler
from wiskers.datasets.trajectories.transforms import EpisodeToTensor


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
    NUM_STEPS = 10
    BATCH_SIZE = 4

    replay_buffer = ReplayBuffer(capacity_episodes=10)
    replay_buffer.add(_make_episode(T=30))

    class DummyCollector:
        min_steps = 10
        max_steps = 50
        policy = None

    sampler = UniformChunkSampler(num_steps=NUM_STEPS)
    transform = EpisodeToTensor(image_size=(3, 8, 8))
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=DummyCollector(),
        transform=transform,
        batch_size=BATCH_SIZE,
    )
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    # Keys
    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}

    # Shapes: (B, T, C, H, W)
    assert batch["obs"].shape == (BATCH_SIZE, NUM_STEPS, 3, 8, 8)
    assert batch["actions"].shape == (BATCH_SIZE, NUM_STEPS, 4)
    assert batch["rewards"].shape == (BATCH_SIZE, NUM_STEPS)
    assert batch["dones"].shape == (BATCH_SIZE, NUM_STEPS)

    # dtypes
    assert batch["obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.float32
    assert batch["rewards"].dtype == torch.float32
    assert batch["dones"].dtype == torch.float32

    # Observations were uint8 [0,255] → normalised to [0,1]
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0


def test_episode_to_tensor_resize_and_grayscale():
    ep = _make_episode(T=6, H=16, W=16, C=3)

    rgb_transform = EpisodeToTensor(image_size=(3, 32, 32))
    out_rgb = rgb_transform(ep)
    assert out_rgb["obs"].shape == (6, 3, 32, 32)
    assert out_rgb["obs"].dtype == torch.float32

    gray_transform = EpisodeToTensor(image_size=(1, 12, 12))
    out_gray = gray_transform(ep)
    assert out_gray["obs"].shape == (6, 1, 12, 12)
    assert out_gray["obs"].dtype == torch.float32

    with pytest.raises(ValueError, match="3-element sequence"):
        EpisodeToTensor(image_size=(64, 64))

    with pytest.raises(ValueError, match="channel count C must be 1 or 3"):
        EpisodeToTensor(image_size=(2, 64, 64))


def test_datamodule_with_collector(monkeypatch):
    class DummyCollector:
        def __init__(self):
            self.min_steps = 10
            self.max_steps = 50
            self.policy = "dummy_policy"
            self.collected = False

        def collect(self, buffer):
            self.collected = True
            buffer.add(_make_episode(T=20))

    collector = DummyCollector()
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=10)
    transform = EpisodeToTensor(image_size=(3, 8, 8))
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
        batch_size=2,
    )
    assert dm.policy == "dummy_policy"
    dm.setup()
    assert collector.collected
    assert dm.replay_buffer.num_episodes == 1


def test_datamodule_collector_validation():
    class DummyCollector:
        def __init__(self, min_steps=10, max_steps=50):
            self.min_steps = min_steps
            self.max_steps = max_steps
            self.policy = None

    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=20)
    transform = EpisodeToTensor(image_size=(3, 8, 8))

    # num_steps > max_steps
    with pytest.raises(ValueError, match="sampler.num_steps .* cannot be greater than collector.max_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=sampler,
            collector=DummyCollector(min_steps=5, max_steps=15),
            transform=transform,
        )

    # collector.min_steps < sampler.num_steps
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=sampler,
            collector=DummyCollector(min_steps=10, max_steps=50),
            transform=transform,
        )
