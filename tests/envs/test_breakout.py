import pytest
import torch
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.datamodule import TrajectoryDataModule
from wiskers.data.sampler import UniformChunkSampler
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.breakout.env import BreakoutEnv
from wiskers.envs.breakout.policies import (
    BreakoutHeuristicPolicy,
    BreakoutNeuralPolicy,
    BreakoutPolicy,
    BreakoutRandomPolicy,
)


def test_breakout_env_observation_shape_and_ram():
    env = BreakoutEnv()
    try:
        assert env._gym_env.unwrapped.ale.getRAMSize() == 128

        obs, _ = env.reset(seed=42)
        assert isinstance(obs, TensorDict)
        assert obs.batch_size == torch.Size([])
        assert obs["render"].shape == (210, 160, 3)
        assert obs["render"].dtype == torch.uint8
        assert obs["ram"].shape == (128,)
        assert obs["ram"].dtype == torch.uint8
        assert torch.equal(obs["ram"], torch.from_numpy(env._gym_env.unwrapped.ale.getRAM()))

        action = env.action_spec.sample()
        assert action.shape == (1,)
        assert action.dtype == torch.int64
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(next_obs, TensorDict)
        assert next_obs.batch_size == torch.Size([])
        assert next_obs["render"].shape == (210, 160, 3)
        assert next_obs["render"].dtype == torch.uint8
        assert next_obs["ram"].shape == (128,)
        assert next_obs["ram"].dtype == torch.uint8
        assert torch.equal(next_obs["ram"], torch.from_numpy(env._gym_env.unwrapped.ale.getRAM()))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_breakout_datamodule_init():
    policy = BreakoutHeuristicPolicy()
    env = BreakoutEnv(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=100)
    sampler = UniformChunkSampler(num_steps=20)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(env=env, controller=policy, transform=transform, num_episodes=0, min_steps=20)
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
    assert dm.controller is policy


def test_breakout_datamodule_validation():
    policy = BreakoutHeuristicPolicy()
    env = BreakoutEnv(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    # sampler.num_steps > collector.max_steps raises ValueError
    with pytest.raises(ValueError, match="sampler.num_steps .* cannot be greater than collector.max_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(
                env=env, controller=policy, transform=transform, num_episodes=1, min_steps=40, max_steps=40
            ),
            transform=transform,
        )

    # collector.min_steps < sampler.num_steps raises ValueError (no silent mutations)
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(
                env=env, controller=policy, transform=transform, num_episodes=1, min_steps=20, max_steps=100
            ),
            transform=transform,
        )

    # default min_steps=0 when sampler.num_steps=50 also raises ValueError
    with pytest.raises(ValueError, match="collector.min_steps .* cannot be less than sampler.num_steps"):
        TrajectoryDataModule(
            replay_buffer=replay_buffer,
            sampler=UniformChunkSampler(num_steps=50),
            collector=Collector(env=env, controller=policy, transform=transform, num_episodes=1, max_steps=100),
            transform=transform,
        )


def test_breakout_datamodule_setup_and_dataloader():
    policy = BreakoutHeuristicPolicy(epsilon=0.0)
    env = BreakoutEnv(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, controller=policy, transform=transform, num_episodes=1, min_steps=5, max_steps=10, verbose=False
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
    assert dm.replay_buffer[0].obs["ram"].shape[1:] == (128,)
    assert dm.replay_buffer[0].obs["ram"].dtype == torch.uint8

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "action", "reward", "done", "terminated", "truncated"}
    assert isinstance(batch["obs"], TensorDict)
    assert batch["obs"]["render"].shape == (2, 6, 3, 64, 64)
    assert batch["action"].shape == (2, 5, 1)
    assert batch["reward"].shape == (2, 5, 1)
    assert batch["done"].shape == (2, 5, 1)
    assert batch["terminated"].shape == (2, 5, 1)
    assert batch["truncated"].shape == (2, 5, 1)
    assert batch["obs"]["render"].dtype == torch.float32
    assert batch["obs"]["render"].min() >= 0.0
    assert batch["obs"]["render"].max() <= 1.0


def test_breakout_datamodule_custom_policy():
    class ConstantFirePolicy(BreakoutPolicy):
        def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
            return torch.tensor([1], dtype=torch.long)

    policy = ConstantFirePolicy()
    env = BreakoutEnv(image_size=(64, 64))
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    sampler = UniformChunkSampler(num_steps=5)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, controller=policy, transform=transform, num_episodes=1, min_steps=5, max_steps=6, verbose=False
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


def test_breakout_random_policy():
    env = BreakoutEnv(image_size=(64, 64))
    policy = BreakoutRandomPolicy(fire_frequency=5)
    obs, _ = env.reset(seed=42)
    assert policy.action_spec.action_dim == 1
    assert int(policy(obs, step=0).item()) == 1
    assert int(policy(obs, step=1).item()) in {0, 1, 2, 3}
    env.close()


def test_breakout_datamodule_with_sampler():
    policy = BreakoutHeuristicPolicy(epsilon=0.0)
    env = BreakoutEnv(image_size=(64, 64))
    sampler = UniformChunkSampler(num_steps=5)
    replay_buffer = ReplayBuffer(capacity_episodes=10)
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env, controller=policy, transform=transform, num_episodes=1, min_steps=5, max_steps=10, verbose=False
    )
    dm = TrajectoryDataModule(
        replay_buffer=replay_buffer,
        sampler=sampler,
        collector=collector,
        transform=transform,
        batch_size=2,
    )
    assert dm.num_steps == 5
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["obs"]["render"].shape == (2, 6, 3, 64, 64)


def test_breakout_neural_policy():
    image_size = (3, 64, 64)
    policy = BreakoutNeuralPolicy(image_size=image_size, hidden_size=32, num_layers=2)

    params = list(policy.parameters())
    assert params and all(p.requires_grad for p in params)

    obs = TensorDict({"render": torch.rand(image_size)}, batch_size=[])
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (1,)
    assert action.dtype == torch.long
    assert int(action.item()) in {0, 1, 2, 3}

    # Backward gradient flow
    feat = obs["render"].flatten()
    logits = policy.actor(feat)
    logits.sum().backward()
    assert all(p.grad is not None for p in params)

    # Eval mode is deterministic
    policy.eval()
    assert torch.equal(policy(obs), policy(obs))


def test_breakout_policies_share_action_contract():
    env = BreakoutEnv(image_size=(64, 64))
    ref_action = env.action_spec.sample()
    obs = TensorDict({"render": torch.rand(3, 64, 64)}, batch_size=[])

    policies = [
        BreakoutRandomPolicy(),
        BreakoutHeuristicPolicy(),
        BreakoutNeuralPolicy(image_size=(3, 64, 64)),
    ]

    for policy in policies:
        assert policy.action_spec == env.action_spec, type(policy).__name__
        policy.reset()
        for step in range(3):
            action = policy(obs, step=step)
            name = type(policy).__name__
            assert isinstance(action, torch.Tensor), name
            assert action.shape == ref_action.shape, f"{name}: {action.shape} != {ref_action.shape}"
            assert action.dtype == ref_action.dtype, f"{name}: {action.dtype} != {ref_action.dtype}"
            assert env.action_spec.contains(action), name

    env.close()
