import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import (
    RecencyWeightedSampler,
    UniformChunkSampler,
)


def _make_episode(T: int = 20, H: int = 8, W: int = 8, C: int = 3, action_dim: int = 4) -> Episode:
    """Build a dummy episode filled with deterministic or pseudo-random data."""
    return Episode(
        obs=np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8),
        actions=np.random.randn(T, action_dim).astype(np.float32),
        rewards=np.random.randn(T).astype(np.float32),
        dones=np.zeros(T, dtype=bool),
    )


def test_uniform_chunk_sampler():
    sampler = UniformChunkSampler(chunk_len=10)
    assert sampler.chunk_len == 10

    buffer = ReplayBuffer(capacity_episodes=10)
    buffer.add(_make_episode(T=25))
    buffer.add(_make_episode(T=30))

    chunks = sampler.sample(buffer, batch_size=4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 10
        assert chunk.obs.shape[0] == 10
        assert chunk.actions.shape[0] == 10


def test_sampler_empty_buffer_raises():
    sampler = UniformChunkSampler(chunk_len=10)
    buffer = ReplayBuffer(capacity_episodes=10)
    with pytest.raises(RuntimeError, match="empty ReplayBuffer"):
        sampler.sample(buffer, batch_size=1)


def test_sampler_no_valid_episodes_raises():
    sampler = UniformChunkSampler(chunk_len=50)
    buffer = ReplayBuffer(capacity_episodes=10)
    buffer.add(_make_episode(T=20))
    with pytest.raises(RuntimeError, match="no episodes in buffer have length >= 50"):
        sampler.sample(buffer, batch_size=1)


def test_recency_weighted_sampler():
    sampler = RecencyWeightedSampler(chunk_len=5, recency_factor=2.0)
    assert sampler.chunk_len == 5
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
        "chunk_len": 35,
    })
    sampler = instantiate(cfg)
    assert isinstance(sampler, UniformChunkSampler)
    assert sampler.chunk_len == 35


def test_datamodule_with_explicit_sampler():
    sampler = UniformChunkSampler(chunk_len=8)
    buffer = ReplayBuffer(capacity_episodes=5)
    buffer.add(_make_episode(T=20))

    dm = TrajectoryDataModule(
        replay_buffer=buffer,
        sampler=sampler,
        batch_size=2,
        steps_per_epoch=2,
    )
    assert dm.chunk_len == 8
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert batch["obs"].shape == (2, 8, 8, 8, 3)
    assert batch["actions"].shape == (2, 8, 4)


def test_replay_buffer_capacity_episodes():
    buf = ReplayBuffer(capacity_episodes=50)
    assert buf.capacity_episodes == 50

