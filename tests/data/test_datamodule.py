import pytest
import torch
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.datamodule import TrajectoryDataModule
from wiskers.data.episode import Episode
from wiskers.data.sampler import UniformChunkSampler
from wiskers.data.transforms import EpisodeToTensor


def _make_episode(T: int = 20, H: int = 8, W: int = 8, C: int = 3, action_dim: int = 4) -> Episode:
    """Build a dummy episode filled with random data."""
    terminated = torch.zeros(T, 1, dtype=torch.bool)
    truncated = torch.zeros(T, 1, dtype=torch.bool)
    return Episode(
        obs=TensorDict({"render": torch.randint(0, 256, (T + 1, H, W, C), dtype=torch.uint8)}, batch_size=[T + 1]),
        action=torch.randn(T, action_dim, dtype=torch.float32),
        reward=torch.randn(T, 1, dtype=torch.float32),
        terminated=terminated,
        truncated=truncated,
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
        controller = None

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
    assert set(batch.keys()) == {"obs", "action", "reward", "done", "terminated", "truncated"}

    # Shapes: obs is TensorDict(B, T + 1) with render (B, T + 1, C, H, W), others are (B, T, ...)
    assert isinstance(batch["obs"], TensorDict)
    assert batch["obs"]["render"].shape == (BATCH_SIZE, NUM_STEPS + 1, 3, 8, 8)
    assert batch["action"].shape == (BATCH_SIZE, NUM_STEPS, 4)
    assert batch["reward"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert batch["done"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert batch["terminated"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert batch["truncated"].shape == (BATCH_SIZE, NUM_STEPS, 1)

    # dtypes
    assert batch["obs"]["render"].dtype == torch.float32
    assert batch["action"].dtype == torch.float32
    assert batch["reward"].dtype == torch.float32
    assert batch["done"].dtype == torch.float32
    assert batch["terminated"].dtype == torch.float32
    assert batch["truncated"].dtype == torch.float32

    # Observations were uint8 [0,255] → normalised to [0,1]
    assert batch["obs"]["render"].min() >= 0.0
    assert batch["obs"]["render"].max() <= 1.0


def test_episode_to_tensor_resize_and_grayscale():
    ep = _make_episode(T=6, H=16, W=16, C=3)

    rgb_transform = EpisodeToTensor(image_size=(3, 32, 32), obs_range=(0.0, 1.0))
    out_rgb = rgb_transform(ep)
    assert isinstance(out_rgb["obs"], TensorDict)
    assert out_rgb["obs"]["render"].shape == (7, 3, 32, 32)
    assert out_rgb["obs"]["render"].dtype == torch.float32
    assert out_rgb["obs"]["render"].min() >= 0.0
    assert out_rgb["obs"]["render"].max() <= 1.0

    gray_transform = EpisodeToTensor(image_size=(1, 12, 12), obs_range=(0.0, 1.0))
    out_gray = gray_transform(ep)
    assert out_gray["obs"]["render"].shape == (7, 1, 12, 12)
    assert out_gray["obs"]["render"].dtype == torch.float32

    # Test obs_range=(-1.0, 1.0)
    terminated = torch.zeros((1, 1), dtype=torch.bool)
    truncated = torch.zeros((1, 1), dtype=torch.bool)
    ep_extremes = Episode(
        obs=TensorDict(
            {"render": torch.tensor([[[[0, 128, 255]]], [[[255, 128, 0]]]], dtype=torch.uint8)},
            batch_size=[2],
        ),
        action=torch.zeros((1, 4), dtype=torch.float32),
        reward=torch.zeros((1, 1), dtype=torch.float32),
        terminated=terminated,
        truncated=truncated,
    )
    sym_transform = EpisodeToTensor(image_size=(3, 1, 1), obs_range=(-1.0, 1.0))
    out_sym = sym_transform(ep_extremes)
    assert torch.isclose(out_sym["obs"]["render"][0, 0, 0, 0], torch.tensor(-1.0))
    assert torch.isclose(out_sym["obs"]["render"][0, 2, 0, 0], torch.tensor(1.0))
    assert -1.0 <= out_sym["obs"]["render"].min() and out_sym["obs"]["render"].max() <= 1.0

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
            self.controller = "dummy_policy"
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
    assert dm.controller == "dummy_policy"
    dm.setup()
    assert collector.collected
    assert dm.replay_buffer.num_episodes == 1


def test_datamodule_collector_validation():
    class DummyCollector:
        def __init__(self, min_steps=10, max_steps=50):
            self.min_steps = min_steps
            self.max_steps = max_steps
            self.controller = None

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
