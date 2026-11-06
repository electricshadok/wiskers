import numpy as np
import pytest
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from wiskers.datasets.trajectories.buffer import ReplayBuffer
from wiskers.datasets.trajectories.collector import Collector
from wiskers.envs.adroit_hand import (
    AVAILABLE_TASKS,
    AdroitHandHeuristicPolicy,
    AdroitHandRandomPolicy,
    make_adroit_hand_env,
)


@pytest.fixture(autouse=True)
def mock_mujoco_renderer(monkeypatch):
    """Bypass OpenGL/GLFW window creation during unit tests while keeping real MuJoCo physics."""
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )


@pytest.fixture
def adroit_env():
    env = make_adroit_hand_env(task="door", image_size=(64, 64))
    yield env
    env.close()


def test_make_adroit_hand_env_observation_shape(adroit_env):
    obs, info = adroit_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8

    action = adroit_env.action_space.sample()
    assert action.shape == (28,)
    next_obs, reward, terminated, truncated, info = adroit_env.step(action)
    assert next_obs.shape == (64, 64, 3)
    assert next_obs.dtype == np.uint8
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_make_adroit_hand_env_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        make_adroit_hand_env(task="non_existent_task")


def test_adroit_hand_available_tasks():
    assert "door" in AVAILABLE_TASKS
    assert "hammer" in AVAILABLE_TASKS
    assert "pen" in AVAILABLE_TASKS
    assert "relocate" in AVAILABLE_TASKS


def test_adroit_random_policy(adroit_env):
    policy = AdroitHandRandomPolicy()
    assert policy.action_type == "continuous"

    obs, _ = adroit_env.reset()
    action = policy(obs, env=adroit_env, step=0)
    assert isinstance(action, np.ndarray)
    assert action.shape == (28,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_adroit_heuristic_policy(adroit_env):
    policy = AdroitHandHeuristicPolicy(momentum=0.8, noise_scale=0.2)
    assert policy.action_type == "continuous"

    obs, _ = adroit_env.reset()
    act1 = policy(obs, env=adroit_env, step=0)
    act2 = policy(obs, env=adroit_env, step=1)
    assert act1.shape == (28,)
    assert act2.shape == (28,)
    assert np.all(act1 >= -1.0) and np.all(act1 <= 1.0)
    assert np.all(act2 >= -1.0) and np.all(act2 <= 1.0)

    policy.reset()
    assert np.all(policy.current_action == 0.0)


def test_adroit_hand_collector_integration(adroit_env):
    policy = AdroitHandHeuristicPolicy()
    collector = Collector(
        env=adroit_env,
        policy=policy,
        num_episodes=1,
        min_steps=5,
        max_steps=10,
        verbose=False,
    )
    buffer = ReplayBuffer(capacity_episodes=5)
    collected_steps = collector.collect(buffer)

    assert collected_steps == 10
    assert buffer.num_episodes == 1
    assert buffer.num_steps == 10
    assert buffer[0].obs.shape == (10, 64, 64, 3)
    assert buffer[0].obs.dtype == np.uint8
    assert buffer[0].actions.shape == (10, 28)
    assert buffer[0].actions.dtype == np.float32

