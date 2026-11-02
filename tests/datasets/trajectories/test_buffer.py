import numpy as np
import pytest

from wiskers.datasets.trajectories.buffer import Buffer, ReplayBuffer, RolloutBuffer
from wiskers.datasets.trajectories.episode import Episode


def _make_episode(T: int = 10, H: int = 4, W: int = 4, C: int = 3, action_dim: int = 2) -> Episode:
    return Episode(
        obs=np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8),
        actions=np.random.randn(T, action_dim).astype(np.float32),
        rewards=np.ones(T, dtype=np.float32),
        dones=np.zeros(T, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Base Buffer Tests
# ---------------------------------------------------------------------------

def test_buffer_base_class():
    # Buffer cannot be instantiated directly
    with pytest.raises(TypeError):
        Buffer()

    replay = ReplayBuffer(capacity_episodes=5)
    rollout = RolloutBuffer()

    assert isinstance(replay, Buffer)
    assert isinstance(rollout, Buffer)
    assert replay.is_empty
    assert rollout.is_empty


# ---------------------------------------------------------------------------
# ReplayBuffer Tests
# ---------------------------------------------------------------------------

def test_replay_buffer_reset():
    buf = ReplayBuffer(capacity_episodes=5)
    buf.add(_make_episode(T=10))
    assert not buf.is_empty
    buf.reset()
    assert buf.is_empty
    assert len(buf) == 0


def test_replay_buffer_add_and_eviction():
    buf = ReplayBuffer(capacity_episodes=2)
    assert buf.is_empty
    assert len(buf) == 0

    ep1 = _make_episode(T=10)
    ep2 = _make_episode(T=20)
    ep3 = _make_episode(T=30)

    buf.add(ep1)
    assert not buf.is_empty
    assert buf.num_episodes == 1
    assert buf.num_steps == 10

    buf.add(ep2)
    assert buf.num_episodes == 2
    assert buf.num_steps == 30

    # Eviction: adding ep3 should evict ep1
    buf.add(ep3)
    assert buf.num_episodes == 2
    assert buf.num_steps == 50
    assert buf[0] is ep2
    assert buf[1] is ep3


def test_replay_buffer_empty_episode_ignored():
    buf = ReplayBuffer(capacity_episodes=5)
    empty_ep = Episode(
        obs=np.empty((0, 4, 4, 3), dtype=np.uint8),
        actions=np.empty((0, 2), dtype=np.float32),
        rewards=np.empty(0, dtype=np.float32),
        dones=np.empty(0, dtype=bool),
    )
    buf.add(empty_ep)
    assert buf.is_empty


def test_replay_buffer_slicing_and_repr():
    buf = ReplayBuffer(capacity_episodes=5)
    for _ in range(3):
        buf.add(_make_episode(T=5))

    sliced = buf[:2]
    assert len(sliced) == 2
    assert "ReplayBuffer(episodes=3/5, total_steps=15)" in repr(buf)


# ---------------------------------------------------------------------------
# RolloutBuffer Tests
# ---------------------------------------------------------------------------

def test_rollout_buffer_add_and_metrics():
    buf = RolloutBuffer()
    assert buf.is_empty
    assert len(buf) == 0

    ep1 = _make_episode(T=5)
    ep2 = _make_episode(T=10)

    buf.add(ep1)
    assert not buf.is_empty
    assert buf.num_episodes == 1
    assert buf.num_steps == 5

    buf.add(ep2)
    assert buf.num_episodes == 2
    assert buf.num_steps == 15
    assert len(buf) == 2
    assert "RolloutBuffer(episodes=2, total_steps=15)" in repr(buf)


def test_rollout_buffer_reset():
    buf = RolloutBuffer()
    buf.add(_make_episode(T=10))
    assert not buf.is_empty
    buf.reset()
    assert buf.is_empty
    assert len(buf) == 0
    assert buf.num_steps == 0


def test_collector_with_rollout_buffer():
    import gymnasium as gym

    from wiskers.datasets.trajectories.collector import Collector

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)
            self.step_count = 0

        def reset(self, seed=None, options=None):
            self.step_count = 0
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action):
            self.step_count += 1
            return np.zeros((4, 4, 3), dtype=np.uint8), 1.0, self.step_count >= 5, False, {}

    collector = Collector(
        env=DummyEnv(),
        policy=lambda obs: 0,
        num_episodes=2,
        min_steps=5,
        max_steps=10,
        verbose=False,
    )
    rollout_buf = RolloutBuffer()
    total_steps = collector.collect(rollout_buf)

    assert total_steps == 10
    assert len(rollout_buf) == 2
    assert rollout_buf.num_episodes == 2

