import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.data.buffer import Buffer, ReplayBuffer, RolloutBuffer
from wiskers.data.episode import Episode


def _make_episode(T: int = 10, H: int = 4, W: int = 4, C: int = 3, action_dim: int = 2) -> Episode:
    terminated = torch.zeros(T, 1, dtype=torch.bool)
    truncated = torch.zeros(T, 1, dtype=torch.bool)
    return Episode(
        obs=TensorDict({"render": torch.randint(0, 256, (T + 1, H, W, C), dtype=torch.uint8)}, batch_size=[T + 1]),
        action=torch.randn(T, action_dim, dtype=torch.float32),
        reward=torch.ones(T, 1, dtype=torch.float32),
        terminated=terminated,
        truncated=truncated,
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
        obs=TensorDict({"render": torch.empty((0, 4, 4, 3), dtype=torch.uint8)}, batch_size=[0]),
        action=torch.empty((0, 2), dtype=torch.float32),
        reward=torch.empty((0, 1), dtype=torch.float32),
        terminated=torch.empty((0, 1), dtype=torch.bool),
        truncated=torch.empty((0, 1), dtype=torch.bool),
    )
    buf.add(empty_ep)
    assert buf.is_empty


def test_episode_validation_and_slice():
    missing_key_td = TensorDict(
        {
            "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
            "action": torch.zeros(5, 2),
            "next": TensorDict(
                {
                    "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
                    "reward": torch.zeros(5, 1),
                    "truncated": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    assert not Episode.is_valid(missing_key_td)
    with pytest.raises(KeyError, match="Episode 'next' missing key: 'done'"):
        Episode.validate(missing_key_td)

    inconsistent_done_td = TensorDict(
        {
            "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
            "action": torch.zeros(5, 2),
            "next": TensorDict(
                {
                    "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
                    "reward": torch.zeros(5, 1),
                    "done": torch.zeros(5, 1, dtype=torch.bool),
                    "terminated": torch.ones(5, 1, dtype=torch.bool),
                    "truncated": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    assert not Episode.is_valid(inconsistent_done_td)
    with pytest.raises(ValueError, match="Episode 'next/done' must equal"):
        Episode.validate(inconsistent_done_td)

    missing_singleton_dim_td = TensorDict(
        {
            "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
            "action": torch.zeros(5, 2),
            "next": TensorDict(
                {
                    "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
                    "reward": torch.zeros(5),  # shape (5,) missing trailing singleton dim
                    "done": torch.zeros(5, 1, dtype=torch.bool),
                    "terminated": torch.zeros(5, 1, dtype=torch.bool),
                    "truncated": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    assert not Episode.is_valid(missing_singleton_dim_td)
    with pytest.raises(ValueError, match="trailing singleton dimension"):
        Episode.validate(missing_singleton_dim_td)

    wrong_reward_dtype_td = TensorDict(
        {
            "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
            "action": torch.zeros(5, 2),
            "next": TensorDict(
                {
                    "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
                    "reward": torch.zeros(5, 1, dtype=torch.int64),  # int instead of float
                    "done": torch.zeros(5, 1, dtype=torch.bool),
                    "terminated": torch.zeros(5, 1, dtype=torch.bool),
                    "truncated": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    assert not Episode.is_valid(wrong_reward_dtype_td)
    with pytest.raises(TypeError, match="floating-point"):
        Episode.validate(wrong_reward_dtype_td)

    wrong_flag_dtype_td = TensorDict(
        {
            "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
            "action": torch.zeros(5, 2),
            "next": TensorDict(
                {
                    "obs": TensorDict({"render": torch.zeros(5, 4, 4, 3)}, batch_size=[5]),
                    "reward": torch.zeros(5, 1, dtype=torch.float32),
                    "done": torch.zeros(5, 1, dtype=torch.float32),  # float instead of bool
                    "terminated": torch.zeros(5, 1, dtype=torch.bool),
                    "truncated": torch.zeros(5, 1, dtype=torch.bool),
                },
                batch_size=[5],
            ),
        },
        batch_size=[5],
    )
    assert not Episode.is_valid(wrong_flag_dtype_td)
    with pytest.raises(TypeError, match="boolean"):
        Episode.validate(wrong_flag_dtype_td)

    ep = _make_episode(T=10)
    assert ep.is_valid()
    assert ep.batch_size == torch.Size([10])
    assert ep.obs.batch_size == torch.Size([10])
    assert ep["next"].batch_size == torch.Size([10])

    # Single transition indexing has batch_size=[]
    trans = ep[0]
    assert trans.batch_size == torch.Size([])
    assert trans["obs"]["render"].shape == (4, 4, 3)
    assert trans["action"].shape == (2,)
    assert trans["next", "reward"].shape == (1,)
    assert trans["next", "done"].shape == (1,)

    chunk = ep.slice(3, 8)
    assert isinstance(chunk, Episode)
    assert len(chunk) == 5
    assert chunk.is_valid()
    assert chunk.obs["render"].shape == (5, 4, 4, 3)
    assert chunk.all_obs("render").shape == (6, 4, 4, 3)
    assert chunk.all_obs().batch_size == torch.Size([6])
    assert chunk.action.shape == (5, 2)
    assert chunk.reward.shape == (5, 1)
    assert chunk.done.shape == (5, 1)
    assert chunk.terminated.shape == (5, 1)
    assert chunk.truncated.shape == (5, 1)
    assert torch.equal(chunk.obs["render"], ep.obs["render"][3:8])
    assert torch.equal(chunk.all_obs("render"), ep.all_obs("render")[3:9])
    assert torch.equal(chunk.action, ep.action[3:8])

    # Multimodal observation support (e.g. state + render)
    multi_modal_ep = Episode(
        obs=TensorDict(
            {
                "state": torch.randn(11, 8),
                "render": torch.randint(0, 256, (11, 4, 4, 3), dtype=torch.uint8),
            },
            batch_size=[11],
        ),
        action=torch.randn(10, 2),
        reward=torch.zeros(10, 1),
        terminated=torch.zeros(10, 1, dtype=torch.bool),
        truncated=torch.zeros(10, 1, dtype=torch.bool),
    )
    assert multi_modal_ep.is_valid()
    assert multi_modal_ep.reward.shape == (10, 1)
    assert multi_modal_ep.all_obs("state").shape == (11, 8)
    assert multi_modal_ep.all_obs("render").shape == (11, 4, 4, 3)

    # obs must have T + 1 steps with explicit batch_size; Episode rejects the mismatch
    with pytest.raises(ValueError, match="obs TensorDict must have explicit batch_size"):
        Episode(
            obs=TensorDict({"render": torch.zeros(10, 4, 4, 3)}, batch_size=[10]),  # 10 instead of 11
            action=torch.randn(10, 2),
            reward=torch.zeros(10, 1),
            terminated=torch.zeros(10, 1, dtype=torch.bool),
            truncated=torch.zeros(10, 1, dtype=torch.bool),
        )

    # Episode rejects unbatched obs
    with pytest.raises(ValueError, match="obs TensorDict must have explicit batch_size"):
        Episode(
            obs=TensorDict({"render": torch.zeros(11, 4, 4, 3)}),  # no batch_size
            action=torch.randn(10, 2),
            reward=torch.zeros(10, 1),
            terminated=torch.zeros(10, 1, dtype=torch.bool),
            truncated=torch.zeros(10, 1, dtype=torch.bool),
        )

    # Episode rejects 1D reward (no backward compatibility auto-unsqueeze)
    with pytest.raises(ValueError, match="reward must be a torch.Tensor with shape"):
        Episode(
            obs=TensorDict({"render": torch.zeros(11, 4, 4, 3)}, batch_size=[11]),
            action=torch.randn(10, 2),
            reward=torch.zeros(10),  # 1D instead of (10, 1)
            terminated=torch.zeros(10, 1, dtype=torch.bool),
            truncated=torch.zeros(10, 1, dtype=torch.bool),
        )


def test_episode_obs_views_share_storage():
    """Ensure obs[:-1] and obs[1:] are zero-copy views sharing the exact same underlying memory storage."""
    raw_frames = torch.randint(0, 256, (11, 16, 16, 3), dtype=torch.uint8)
    obs = TensorDict({"render": raw_frames}, batch_size=[11])
    ep = Episode(
        obs=obs,
        action=torch.randn(10, 2),
        reward=torch.zeros(10, 1),
        terminated=torch.zeros(10, 1, dtype=torch.bool),
        truncated=torch.zeros(10, 1, dtype=torch.bool),
    )

    curr_ptr = ep["obs", "render"].untyped_storage().data_ptr()
    next_ptr = ep["next", "obs", "render"].untyped_storage().data_ptr()
    raw_ptr = raw_frames.untyped_storage().data_ptr()

    # Zero-copy verification: all views must point to the identical data pointer
    assert curr_ptr == raw_ptr, f"curr_ptr ({curr_ptr}) != raw_ptr ({raw_ptr})"
    assert next_ptr == raw_ptr, f"next_ptr ({next_ptr}) != raw_ptr ({raw_ptr})"
    assert curr_ptr == next_ptr, f"curr_ptr ({curr_ptr}) != next_ptr ({next_ptr})"


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
    import torch

    from wiskers.data.collector import Collector
    from wiskers.data.transforms import EpisodeToTensor
    from wiskers.envs import TorchGymEnv
    from wiskers.policies import DiscretePolicy

    class DummyPolicy(DiscretePolicy):
        def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
            return torch.tensor([0], dtype=torch.long)

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
        env=TorchGymEnv(DummyEnv()),
        policy=DummyPolicy(),
        transform=EpisodeToTensor(image_size=(3, 4, 4), obs_range=(0.0, 1.0)),
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
