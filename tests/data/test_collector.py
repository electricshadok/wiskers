import gymnasium as gym
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.policy.base import ContinuousPolicy, DiscretePolicy


class DummyPolicy(DiscretePolicy):
    def __init__(self) -> None:
        super().__init__(DiscreteActionSpec(n=4))

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        assert isinstance(obs, TensorDict)
        return torch.tensor([0], dtype=torch.long)


class DummyContinuousPolicy(ContinuousPolicy):
    def __init__(self) -> None:
        super().__init__(
            BoundedActionSpec(
                low=torch.full((2,), -1.0, dtype=torch.float32),
                high=torch.full((2,), 1.0, dtype=torch.float32),
            )
        )

    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        assert isinstance(obs, TensorDict)
        return torch.tensor([0.25, -0.75], dtype=torch.float32)


class DummyDiscreteEnv(gym.Env):
    def __init__(self, step_limit: int = 10):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8)
        self.step_limit = step_limit
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.zeros((16, 16, 3), dtype=np.uint8), {}

    def step(self, action):
        self.current_step += 1
        obs = np.full((16, 16, 3), self.current_step, dtype=np.uint8)
        reward = 1.0
        terminated = self.current_step >= self.step_limit
        truncated = False
        return obs, reward, terminated, truncated, {}


class DummyContinuousEnv(gym.Env):
    def __init__(self, step_limit: int = 15):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (16, 16, 3), dtype=np.uint8)
        self.step_limit = step_limit
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.ones((16, 16, 3), dtype=np.uint8), {}

    def step(self, action):
        self.current_step += 1
        obs = np.ones((16, 16, 3), dtype=np.uint8)
        reward = 0.5
        terminated = self.current_step >= self.step_limit
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_collector_collect_one_episode_no_args():
    """collect_one_episode takes no parameters and uses self.env, self.policy, self.max_steps."""
    env = TorchGymEnv(DummyDiscreteEnv(step_limit=8))
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(env=env, controller=DummyPolicy(), transform=transform, max_steps=20, verbose=False)
    episode = collector.collect_one_episode()

    assert len(episode) == 8
    assert episode.is_valid()
    assert episode.obs["render"].dtype == torch.uint8
    assert episode.obs["render"].shape == (8, 16, 16, 3)
    assert episode["next", "obs", "render"].shape == (8, 16, 16, 3)
    # Initial observation o_0 is zeros, terminal observation o_8 is 8s
    assert torch.all(episode.obs["render"][0] == 0)
    assert torch.all(episode["next", "obs", "render"][-1] == 8)
    assert episode.action.shape == (8, 1)
    assert episode.action.dtype == torch.int64
    assert episode.reward.shape == (8, 1)
    assert episode.done.shape == (8, 1)
    assert episode.terminated.shape == (8, 1)
    assert episode.truncated.shape == (8, 1)
    assert episode.terminated[-1].item() is True
    assert episode.terminated[:-1].any().item() is False
    assert episode.truncated.any().item() is False
    assert torch.equal(episode.done, episode.terminated | episode.truncated)


def test_collector_terminated_and_truncated_distinction():
    """Verify that Gymnasium terminated and truncated flags flow accurately into Episode."""

    class TruncatedEnv(gym.Env):
        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(4)
            self.step_count = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            return np.zeros((16, 16, 3), dtype=np.uint8), {}

        def step(self, action):
            self.step_count += 1
            terminated = False
            truncated = self.step_count >= 5
            return np.ones((16, 16, 3), dtype=np.uint8), 1.0, terminated, truncated, {}

    env = TorchGymEnv(TruncatedEnv())
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(env=env, controller=DummyPolicy(), transform=transform, max_steps=20, verbose=False)
    episode = collector.collect_one_episode()

    assert len(episode) == 5
    assert episode.terminated.any().item() is False
    assert episode.truncated[-1].item() is True
    assert episode.truncated[:-1].any().item() is False
    assert episode.done[-1].item() is True
    assert torch.equal(episode.done, episode.terminated | episode.truncated)


def test_collector_max_steps_clamping():
    """Collector clamps rollouts at max_steps if environment does not terminate earlier."""
    env = TorchGymEnv(DummyDiscreteEnv(step_limit=50))
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(env=env, controller=DummyPolicy(), transform=transform, max_steps=12, verbose=False)
    episode = collector.collect_one_episode()

    assert len(episode) == 12


def test_collector_min_steps_filtering():
    """Episodes shorter than min_steps are discarded until num_episodes valid ones are collected."""

    class FlakyEnv(DummyDiscreteEnv):
        def __init__(self):
            super().__init__()
            self.resets = 0

        def reset(self, seed=None, options=None):
            self.resets += 1
            # Every other episode terminates early at 3 steps, valid episodes terminate at 15 steps
            self.step_limit = 3 if self.resets % 2 == 1 else 15
            return super().reset(seed=seed)

    env = TorchGymEnv(FlakyEnv())
    buffer = ReplayBuffer(capacity_episodes=10)
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env,
        controller=DummyPolicy(),
        transform=transform,
        num_episodes=3,
        min_steps=10,  # Must reject 3-step episodes
        max_steps=50,
        verbose=False,
    )
    total_steps = collector.collect(buffer)

    assert buffer.num_episodes == 3
    assert total_steps == 45  # 3 * 15
    for ep in buffer.episodes:
        assert len(ep) >= 10


def test_collector_continuous_actions():
    """Collector properly formats continuous array actions."""
    env = TorchGymEnv(DummyContinuousEnv(step_limit=6))
    buffer = ReplayBuffer(capacity_episodes=5)
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(
        env=env,
        controller=DummyContinuousPolicy(),
        transform=transform,
        num_episodes=2,
        verbose=False,
    )
    total_steps = collector.collect(buffer)

    assert total_steps == 12
    assert buffer.num_episodes == 2
    assert buffer[0].action.shape == (6, 2)
    assert buffer[0].action.dtype == torch.float32


def test_collector_collect_one_episode_with_idx():
    """collect_one_episode accepts an explicit episode_idx parameter."""
    env = TorchGymEnv(DummyDiscreteEnv(step_limit=4))
    transform = EpisodeToTensor(image_size=(3, 16, 16), obs_range=(0.0, 1.0))
    collector = Collector(env=env, controller=DummyPolicy(), transform=transform, seed=10, max_steps=20, verbose=False)
    ep0 = collector.collect_one_episode(episode_idx=0)
    ep5 = collector.collect_one_episode(episode_idx=5)
    assert len(ep0) == 4
    assert len(ep5) == 4


def test_collector_hydra_instantiation():
    """Collector can be instantiated via Hydra configuration with _target_."""
    cfg = OmegaConf.create(
        {
            "_target_": "wiskers.data.collector.Collector",
            "env": {
                "_target_": "wiskers.envs.torch_env.TorchGymEnv",
                "env": {
                    "_target_": "tests.data.test_collector.DummyDiscreteEnv",
                    "step_limit": 20,
                },
            },
            "controller": {
                "_target_": "tests.data.test_collector.DummyPolicy",
            },
            "transform": {
                "_target_": "wiskers.data.transforms.EpisodeToTensor",
                "image_size": [3, 16, 16],
                "obs_range": [0.0, 1.0],
            },
            "num_episodes": 25,
            "min_steps": 50,
            "max_steps": 500,
        }
    )
    collector = instantiate(cfg)
    assert isinstance(collector, Collector)
    assert isinstance(collector.env, TorchGymEnv)
    assert collector.num_episodes == 25
    assert collector.min_steps == 50
    assert collector.max_steps == 500


def test_collector_with_transform():
    """Collector passes transformed TensorDict obs to controller while storing uint8 tensor in Episode."""
    seen_obs: list[TensorDict] = []

    class InspectingPolicy(DiscretePolicy):
        def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
            seen_obs.append(obs)
            return torch.tensor([0], dtype=torch.long)

    env = TorchGymEnv(DummyDiscreteEnv(step_limit=4))
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(-1.0, 1.0))
    collector = Collector(
        env=env,
        controller=InspectingPolicy(DiscreteActionSpec(n=4)),
        transform=transform,
        max_steps=10,
        verbose=False,
    )
    ep = collector.collect_one_episode()

    assert len(seen_obs) == 4
    assert isinstance(seen_obs[0], TensorDict)
    assert seen_obs[0]["render"].shape == (3, 8, 8)
    assert seen_obs[0]["render"].dtype == torch.float32
    assert torch.allclose(seen_obs[0]["render"], torch.full((3, 8, 8), -1.0))
    # Episode stores raw uint8 transitions [T, H, W, C]
    assert ep.obs["render"].dtype == torch.uint8
    assert ep.obs["render"].shape == (4, 16, 16, 3)
    assert ep["next", "obs", "render"].shape == (4, 16, 16, 3)


def test_torch_gym_env_handles_negative_stride_observations():
    """TorchGymEnv converts negative-stride observations (e.g., MuJoCo flipped OpenGL views) cleanly."""

    class FlippedObsEnv(DummyDiscreteEnv):
        def reset(self, *, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            return obs[::-1, :, :], info

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return obs[::-1, :, :], reward, terminated, truncated, info

    env = TorchGymEnv(FlippedObsEnv(step_limit=3))
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, TensorDict)
    assert obs["render"].dtype == torch.uint8
    assert obs["render"].shape == (16, 16, 3)

    next_obs, _, _, _, _ = env.step(torch.tensor(0))
    assert isinstance(next_obs, TensorDict)
    assert next_obs["render"].dtype == torch.uint8
    assert next_obs["render"].shape == (16, 16, 3)
