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
        obs=torch.randint(0, 256, (T + 1, H, W, C), dtype=torch.uint8),
        actions=torch.randn(T, action_dim, dtype=torch.float32),
        rewards=torch.randn(T, dtype=torch.float32),
        dones=torch.zeros(T, dtype=torch.bool),
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
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0))
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

    # Shapes: obs is (B, T + 1, C, H, W), others are (B, T, ...)
    assert batch["obs"].shape == (BATCH_SIZE, NUM_STEPS + 1, 3, 8, 8)
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

    rgb_transform = EpisodeToTensor(image_size=(3, 32, 32), obs_range=(0.0, 1.0))
    out_rgb = rgb_transform(ep)
    assert out_rgb["obs"].shape == (7, 3, 32, 32)
    assert out_rgb["obs"].dtype == torch.float32
    assert out_rgb["obs"].min() >= 0.0
    assert out_rgb["obs"].max() <= 1.0

    gray_transform = EpisodeToTensor(image_size=(1, 12, 12), obs_range=(0.0, 1.0))
    out_gray = gray_transform(ep)
    assert out_gray["obs"].shape == (7, 1, 12, 12)
    assert out_gray["obs"].dtype == torch.float32

    # Test obs_range=(-1.0, 1.0)
    ep_extremes = Episode(
        obs=torch.tensor([[[[0, 128, 255]]], [[[255, 128, 0]]]], dtype=torch.uint8),
        actions=torch.zeros((1, 4), dtype=torch.float32),
        rewards=torch.zeros((1,), dtype=torch.float32),
        dones=torch.zeros((1,), dtype=torch.bool),
    )
    sym_transform = EpisodeToTensor(image_size=(3, 1, 1), obs_range=(-1.0, 1.0))
    out_sym = sym_transform(ep_extremes)
    assert torch.isclose(out_sym["obs"][0, 0, 0, 0], torch.tensor(-1.0))
    assert torch.isclose(out_sym["obs"][0, 2, 0, 0], torch.tensor(1.0))
    assert -1.0 <= out_sym["obs"].min() and out_sym["obs"].max() <= 1.0

    with pytest.raises(ValueError, match="3-element sequence"):
        EpisodeToTensor(image_size=(64, 64), obs_range=(0.0, 1.0))

    with pytest.raises(ValueError, match="channel count C must be 1 or 3"):
        EpisodeToTensor(image_size=(2, 64, 64), obs_range=(0.0, 1.0))

    with pytest.raises(ValueError, match="2-element sequence"):
        EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 0.5, 1.0))

    with pytest.raises(ValueError, match="low < high"):
        EpisodeToTensor(image_size=(3, 64, 64), obs_range=(1.0, -1.0))


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
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0))
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
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0))

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
