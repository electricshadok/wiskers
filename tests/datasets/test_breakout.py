import torch

from wiskers.datasets.gym.breakout import BreakoutDataModule
from wiskers.datasets.gym.breakout_policies import (
    BreakoutHeuristicPolicy,
    BreakoutPolicy,
    BreakoutRandomPolicy,
)
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


def test_breakout_datamodule_init():
    policy = BreakoutHeuristicPolicy()
    replay_buffer = ReplayBuffer(capacity=100, chunk_len=20)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        policy=policy,
        seed_episodes=0,
        batch_size=4,
        steps_per_epoch=4,
    )
    assert dm.replay_buffer.capacity == 100
    assert dm.replay_buffer.chunk_len == 20
    assert dm.replay_buffer.is_empty
    assert dm.policy is policy


def test_breakout_datamodule_setup_and_dataloader():
    replay_buffer = ReplayBuffer(capacity=10, chunk_len=5)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        policy=BreakoutHeuristicPolicy(epsilon=0.0),
        seed_episodes=1,
        max_steps_per_episode=10,
        image_size=(64, 64),
        batch_size=2,
        steps_per_epoch=2,
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

    replay_buffer = ReplayBuffer(capacity=10, chunk_len=5)
    dm = BreakoutDataModule(
        replay_buffer=replay_buffer,
        policy=ConstantFirePolicy(),
        seed_episodes=1,
        max_steps_per_episode=6,
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
