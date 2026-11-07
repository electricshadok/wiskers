import gymnasium as gym
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.collector import Collector


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
    env = DummyDiscreteEnv(step_limit=8)

    def policy(obs, env, step):
        return 0

    collector = Collector(env=env, policy=policy, max_steps=20, verbose=False)
    episode = collector.collect_one_episode()

    assert len(episode) == 8
    assert episode.obs.dtype == np.uint8
    assert episode.obs.shape == (9, 16, 16, 3)
    # Initial observation o_0 is zeros, terminal observation o_8 is 8s
    assert np.all(episode.obs[0] == 0)
    assert np.all(episode.obs[8] == 8)
    assert episode.actions.shape == (8, 1)
    assert episode.actions.dtype == np.float32


def test_collector_max_steps_clamping():
    """Collector clamps rollouts at max_steps if environment does not terminate earlier."""
    env = DummyDiscreteEnv(step_limit=50)

    def policy(obs, env, step):
        return 1

    collector = Collector(env=env, policy=policy, max_steps=12, verbose=False)
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

    env = FlakyEnv()

    def policy(obs, env, step):
        return 0

    buffer = ReplayBuffer(capacity_episodes=10)
    collector = Collector(
        env=env,
        policy=policy,
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
    env = DummyContinuousEnv(step_limit=6)

    def policy(obs, env, step):
        return np.array([0.25, -0.75], dtype=np.float32)

    buffer = ReplayBuffer(capacity_episodes=5)
    collector = Collector(env=env, policy=policy, num_episodes=2, verbose=False)
    total_steps = collector.collect(buffer)

    assert total_steps == 12
    assert buffer.num_episodes == 2
    assert buffer[0].actions.shape == (6, 2)
    assert buffer[0].actions.dtype == np.float32


class DummyPolicy:
    def __call__(self, obs, env=None, step=0):
        return 0


def test_collector_hydra_instantiation():
    """Collector can be instantiated via Hydra configuration with _target_."""
    cfg = OmegaConf.create({
        "_target_": "wiskers.datasets.trajectories.collector.Collector",
        "env": {
            "_target_": "tests.datasets.trajectories.test_collector.DummyDiscreteEnv",
            "step_limit": 20,
        },
        "policy": {
            "_target_": "tests.datasets.trajectories.test_collector.DummyPolicy",
        },
        "num_episodes": 25,
        "min_steps": 50,
        "max_steps": 500,
    })
    collector = instantiate(cfg)
    assert isinstance(collector, Collector)
    assert collector.num_episodes == 25
    assert collector.min_steps == 50
    assert collector.max_steps == 500
