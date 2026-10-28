import torch

from wiskers.datasets.gym.breakout import BreakoutDataModule
from wiskers.datasets.gym.breakout_policies import (
    BreakoutHeuristicPolicy,
    BreakoutPolicy,
    BreakoutRandomPolicy,
)
from wiskers.datasets.trajectories.config import CollectionConfig, TrainingConfig
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer
from wiskers.datasets.trajectories.sampler import UniformChunkSampler


def test_breakout_datamodule_init():
    policy = BreakoutHeuristicPolicy()
    replay_buffer = ReplayBuffer(capacity_episodes=100)
    sampler = UniformChunkSampler(chunk_len=20)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        policy=policy,
        collection=CollectionConfig(seed_episodes=0),
        training=TrainingConfig(batch_size=4, steps_per_epoch=4),
    )
    assert dm.replay_buffer.capacity_episodes == 100
    assert dm.chunk_len == 20
    assert dm.replay_buffer.is_empty
    assert dm.policy is policy


def test_breakout_datamodule_setup_and_dataloader():
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(chunk_len=5)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        policy=BreakoutHeuristicPolicy(epsilon=0.0),
        collection=CollectionConfig(seed_episodes=1, max_steps_per_episode=10),
        training=TrainingConfig(batch_size=2, steps_per_epoch=2),
        image_size=(64, 64),
        verbose=False,
    )
    dm.setup()
    assert not dm.replay_buffer.is_empty
    assert dm.replay_buffer.num_episodes == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}
    assert batch["obs"].shape == (2, 5, 64, 64, 3)
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

    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(chunk_len=5)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        policy=ConstantFirePolicy(),
        collection=CollectionConfig(seed_episodes=1, max_steps_per_episode=6),
        training=TrainingConfig(batch_size=2, steps_per_epoch=2),
        verbose=False,
    )
    dm.setup()
    assert dm.replay_buffer.num_episodes == 1


def test_breakout_random_policy():
    policy = BreakoutRandomPolicy(fire_frequency=5)
    assert policy.action_dim == 1
    assert policy.action_type == "discrete"
    assert policy(None, None, step=0) == 1
    assert policy(None, None, step=1) in {0, 1, 2, 3}


def test_breakout_datamodule_with_sampler():
    sampler = UniformChunkSampler(chunk_len=5)
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        policy=BreakoutHeuristicPolicy(epsilon=0.0),
        collection=CollectionConfig(seed_episodes=1, max_steps_per_episode=10),
        training=TrainingConfig(batch_size=2, steps_per_epoch=2),
        image_size=(64, 64),
        verbose=False,
    )
    assert dm.chunk_len == 5
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["obs"].shape == (2, 5, 64, 64, 3)

