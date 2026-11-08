import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.sampler import (
    RecencyWeightedSampler,
    UniformChunkSampler,
)
from wiskers.datasets.trajectories.transforms import EpisodeToTensor


def _make_episode(T: int = 20, H: int = 8, W: int = 8, C: int = 3, action_dim: int = 4) -> Episode:
    """Build a dummy episode filled with deterministic or pseudo-random data."""
    return Episode(
        obs=torch.randint(0, 256, (T + 1, H, W, C), dtype=torch.uint8),
        actions=torch.randn(T, action_dim, dtype=torch.float32),
        rewards=torch.randn(T, dtype=torch.float32),
        dones=torch.zeros(T, dtype=torch.bool),
    )


def test_uniform_chunk_sampler():
    sampler = UniformChunkSampler(num_steps=10)
    assert sampler.num_steps == 10

    buffer = ReplayBuffer(capacity_episodes=10)
    buffer.add(_make_episode(T=25))
    buffer.add(_make_episode(T=30))

    chunks = sampler.sample(buffer, batch_size=4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 10
        assert chunk.obs.shape[0] == 11
        assert chunk.actions.shape[0] == 10


def test_sampler_empty_buffer_raises():
    sampler = UniformChunkSampler(num_steps=10)
    buffer = ReplayBuffer(capacity_episodes=10)
    with pytest.raises(RuntimeError, match="empty ReplayBuffer"):
        sampler.sample(buffer, batch_size=1)


def test_sampler_no_valid_episodes_raises():
    sampler = UniformChunkSampler(num_steps=50)
    buffer = ReplayBuffer(capacity_episodes=10)
    buffer.add(_make_episode(T=20))
    with pytest.raises(RuntimeError, match="no episodes in buffer have length >= 50"):
        sampler.sample(buffer, batch_size=1)


def test_recency_weighted_sampler():
    sampler = RecencyWeightedSampler(num_steps=5, recency_factor=2.0)
    assert sampler.num_steps == 5
    assert sampler.recency_factor == 2.0

    buffer = ReplayBuffer(capacity_episodes=10)
    for _ in range(5):
        buffer.add(_make_episode(T=15))

    chunks = sampler.sample(buffer, batch_size=4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 5


def test_hydra_sampler_instantiation():
    cfg = OmegaConf.create({
        "_target_": "wiskers.datasets.trajectories.sampler.UniformChunkSampler",
        "num_steps": 35,
    })
    sampler = instantiate(cfg)
    assert isinstance(sampler, UniformChunkSampler)
    assert sampler.num_steps == 35


def test_datamodule_with_explicit_sampler():
    sampler = UniformChunkSampler(num_steps=8)
    buffer = ReplayBuffer(capacity_episodes=5)
    buffer.add(_make_episode(T=20))

    class DummyCollector:
        min_steps = 8
        max_steps = 50
        policy = None

    dm = TrajectoryDataModule(
        replay_buffer=buffer,
        sampler=sampler,
        collector=DummyCollector(),
        transform=EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0)),
        batch_size=2,
    )
    assert dm.num_steps == 8
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch["obs"].shape == (2, 9, 3, 8, 8)
    assert batch["actions"].shape == (2, 8, 4)


def test_replay_buffer_capacity_episodes():
    buf = ReplayBuffer(capacity_episodes=50)
    assert buf.capacity_episodes == 50

