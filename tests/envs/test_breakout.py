import pytest
import torch

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.sampler import UniformChunkSampler
from wiskers.datasets.trajectories.transforms import EpisodeToTensor
from wiskers.envs.breakout import (
    BreakoutHeuristicPolicy,
    BreakoutPolicy,
    BreakoutRandomPolicy,
    make_breakout_env,
)


def test_breakout_datamodule_init():
    policy = BreakoutHeuristicPolicy()
    env = make_breakout_env()
    replay_buffer = ReplayBuffer(capacity_episodes=100)
    sampler = UniformChunkSampler(num_steps=20)
    collector = Collector(env=env, policy=policy, num_episodes=0, min_steps=20)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0)),
        batch_size=4,
    )
    assert dm.replay_buffer.capacity_episodes == 100
    assert dm.num_steps == 20
    assert dm.collector.min_steps == 20
    assert dm.replay_buffer.is_empty
    assert dm.policy is policy


def test_breakout_datamodule_validation():
    policy = BreakoutHeuristicPolicy()
    env = make_breakout_env()
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    # sampler.num_steps > collector.max_steps raises ValueError
    with pytest.raises(ValueError, match="sampler.num_steps .* cannot be greater than collector.max_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, min_steps=40, max_steps=40),
            transform=transform,
        )

    # collector.min_steps < sampler.num_steps raises ValueError (no silent mutations)
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, min_steps=20, max_steps=100),
            transform=transform,
        )

    # default min_steps=0 when sampler.num_steps=50 also raises ValueError
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, max_steps=100),
            transform=transform,
        )


def test_breakout_datamodule_setup_and_dataloader():
    policy = BreakoutHeuristicPolicy(epsilon=0.0)
    env = make_breakout_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=5, max_steps=10, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0)),
        batch_size=2,
    )
    dm.setup()
    assert not dm.replay_buffer.is_empty
    assert dm.replay_buffer.num_episodes == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}
    assert batch["obs"].shape == (2, 6, 3, 64, 64)
    assert batch["actions"].shape == (2, 5, 1)
    assert batch["rewards"].shape == (2, 5)
    assert batch["dones"].shape == (2, 5)
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0


def test_breakout_datamodule_custom_policy():
    class ConstantFirePolicy(BreakoutPolicy):
        def forward(self, obs, env=None, step=0):
            return 1

    policy = ConstantFirePolicy()
    env = make_breakout_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=5, max_steps=6, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0)),
        batch_size=2,
    )
    dm.setup()
    assert dm.replay_buffer.num_episodes == 1


def test_breakout_random_policy():
    env = make_breakout_env(image_size=(64, 64))
    policy = BreakoutRandomPolicy(fire_frequency=5)
    assert policy.action_dim == 1
    assert policy.action_type == "discrete"
    assert policy(None, env, step=0) == 1
    assert policy(None, env, step=1) in {0, 1, 2, 3}
    env.close()


def test_breakout_datamodule_with_sampler():
    policy = BreakoutHeuristicPolicy(epsilon=0.0)
    env = make_breakout_env(image_size=(64, 64))
    sampler = UniformChunkSampler(num_steps=5)
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=5, max_steps=10, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0)),
        batch_size=2,
    )
    assert dm.num_steps == 5
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["obs"].shape == (2, 6, 3, 64, 64)
