import numpy as np
import pytest
import torch

from wiskers.datasets.gym.car_racing_collector import make_car_racing_env
from wiskers.datasets.gym.car_racing_policies import (
    CarRacingHeuristicPolicy,
    CarRacingPolicy,
    CarRacingRandomPolicy,
)
from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import UniformChunkSampler


def test_car_racing_datamodule_init():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env()
    replay_buffer = ReplayBuffer(capacity_episodes=100)
    sampler = UniformChunkSampler(num_steps=20)
    collector = Collector(env=env, policy=policy, num_episodes=0, min_steps=20)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        batch_size=4,
    )
    assert dm.replay_buffer.capacity_episodes == 100
    assert dm.num_steps == 20
    assert dm.collector.min_steps == 20
    assert dm.replay_buffer.is_empty
    assert dm.policy is policy


def test_car_racing_datamodule_validation():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env()
    replay_buffer = ReplayBuffer(capacity_episodes=10)

    # sampler.num_steps > collector.max_steps raises ValueError
    with pytest.raises(ValueError, match="sampler.num_steps .* cannot be greater than collector.max_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, min_steps=40, max_steps=40),
        )

    # collector.min_steps < sampler.num_steps raises ValueError (no silent mutations)
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, min_steps=20, max_steps=100),
        )

    # default min_steps=0 when sampler.num_steps=50 also raises ValueError
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, num_episodes=1, max_steps=100),
        )


def test_car_racing_datamodule_setup_and_dataloader():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=5, max_steps=8, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        batch_size=2,
    )
    dm.setup()
    assert not dm.replay_buffer.is_empty
    assert dm.replay_buffer.num_episodes == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}
    assert batch["obs"].shape == (2, 5, 3, 64, 64)
    assert batch["actions"].shape == (2, 5, 3)
    assert batch["rewards"].shape == (2, 5)
    assert batch["dones"].shape == (2, 5)
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0


def test_car_racing_datamodule_custom_policy():
    class StraightLinePolicy(CarRacingPolicy):
        def forward(self, obs, env=None, step=0):
            return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    policy = StraightLinePolicy()
    env = make_car_racing_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=5, max_steps=6, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        batch_size=2,
    )
    dm.setup()
    assert dm.replay_buffer.num_episodes == 1


def test_car_racing_heuristic_policy():
    policy = CarRacingHeuristicPolicy()
    assert policy.action_dim == 3
    assert policy.action_type == "continuous"
    policy.steering = 0.5
    policy.reset()
    assert policy.steering == 0.0
    action = policy(None, None, step=0)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)


def test_car_racing_random_policy():
    policy = CarRacingRandomPolicy()
    action = policy(None, None, step=0)
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)


def test_car_racing_datamodule_native_resolution():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env(image_size=None)
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=4)
    collector = Collector(env=env, policy=policy, num_episodes=1, min_steps=4, max_steps=6, verbose=False)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        batch_size=2,
    )
    assert dm.num_steps == 4
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["obs"].shape == (2, 4, 3, 96, 96)
