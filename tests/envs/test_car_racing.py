import pytest
import torch

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule
from wiskers.datasets.trajectories.sampler import UniformChunkSampler
from wiskers.datasets.trajectories.transforms import EpisodeToTensor
from wiskers.envs.car_racing import (
    CarRacingHeuristicPolicy,
    CarRacingPolicy,
    CarRacingRandomPolicy,
    make_car_racing_env,
)


def test_car_racing_datamodule_init():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env()
    replay_buffer = ReplayBuffer(capacity_episodes=100)
    sampler = UniformChunkSampler(num_steps=20)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(env=env, policy=policy, transform=transform, num_episodes=0, min_steps=20)
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
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
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    # sampler.num_steps > collector.max_steps raises ValueError
    with pytest.raises(ValueError, match="sampler.num_steps .* cannot be greater than collector.max_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(
                env=env, policy=policy, transform=transform, num_episodes=1, min_steps=40, max_steps=40
            ),
            transform=transform,
        )

    # collector.min_steps < sampler.num_steps raises ValueError (no silent mutations)
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(
                env=env, policy=policy, transform=transform, num_episodes=1, min_steps=20, max_steps=100
            ),
            transform=transform,
        )

    # default min_steps=0 when sampler.num_steps=50 also raises ValueError
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, policy=policy, transform=transform, num_episodes=1, max_steps=100),
            transform=transform,
        )


def test_car_racing_datamodule_setup_and_dataloader():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, policy=policy, transform=transform, num_episodes=1, min_steps=5, max_steps=8, verbose=False
    )
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
        batch_size=2,
    )
    dm.setup()
    assert not dm.replay_buffer.is_empty
    assert dm.replay_buffer.num_episodes == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}
    assert batch["obs"].shape == (2, 6, 3, 64, 64)
    assert batch["actions"].shape == (2, 5, 3)
    assert batch["rewards"].shape == (2, 5)
    assert batch["dones"].shape == (2, 5)
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0


def test_car_racing_datamodule_custom_policy():
    class StraightLinePolicy(CarRacingPolicy):
        def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
            return torch.tensor([0.0, 0.5, 0.0], dtype=torch.float32)

    policy = StraightLinePolicy()
    env = make_car_racing_env(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, policy=policy, transform=transform, num_episodes=1, min_steps=5, max_steps=6, verbose=False
    )
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
        batch_size=2,
    )
    dm.setup()
    assert dm.replay_buffer.num_episodes == 1


def test_car_racing_heuristic_policy():
    env = make_car_racing_env(image_size=(64, 64))
    policy = CarRacingHeuristicPolicy(env=env)
    obs, _ = env.reset()
    obs_t = torch.from_numpy(obs)
    assert policy.action_dim == 3
    policy.steering = 0.5
    policy.reset()
    assert policy.steering == 0.0
    action = policy(obs_t, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (3,)
    env.close()


def test_car_racing_random_policy():
    env = make_car_racing_env(image_size=(64, 64))
    policy = CarRacingRandomPolicy(env=env)
    obs, _ = env.reset()
    obs_t = torch.from_numpy(obs)
    action = policy(obs_t, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (3,)
    env.close()


def test_car_racing_datamodule_native_resolution():
    policy = CarRacingHeuristicPolicy()
    env = make_car_racing_env(image_size=None)
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=4)
    transform = EpisodeToTensor(image_size=(3, 96, 96), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, policy=policy, transform=transform, num_episodes=1, min_steps=4, max_steps=6, verbose=False
    )
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
        batch_size=2,
    )
    assert dm.num_steps == 4
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["obs"].shape == (2, 5, 3, 96, 96)
