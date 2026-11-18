import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.controller import Controller
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.planner.base import DynamicsModel, Planner


class DummyDynamicsModel:
    """Minimal toy dynamics model recording encode_step calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor | None, int | None]] = []

    def encode_step(
        self,
        obs: TensorDict,
        prev_action: torch.Tensor | None = None,
        state: int | None = None,
    ) -> int:
        self.calls.append((prev_action, state))
        return 1 if state is None else state + 1

    def imagine_returns(self, state: int, actions: torch.Tensor) -> torch.Tensor:
        return -(actions**2).sum(dim=(-1, -2))


class ZeroPlanner(Planner):
    """Concrete minimal Planner returning zeros."""

    def plan(self, state: int) -> torch.Tensor:
        return torch.zeros(self.action_dim, dtype=torch.float32)


class _DummyBoxEnv(gym.Env):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((8, 8, 3), dtype=np.uint8), 0.0, False, False, {}


def test_planner_is_abstract():
    model = DummyDynamicsModel()
    assert isinstance(model, DynamicsModel)
    with pytest.raises(TypeError):
        Planner(model=model, action_dim=2)  # type: ignore[abstract]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_dim": 0},
        {"action_dim": 2, "horizon": 0},
        {"action_dim": 2, "num_samples": 0},
        {"action_dim": 2, "low": 1.0, "high": 1.0},
    ],
)
def test_planner_invalid_args(kwargs):
    with pytest.raises(ValueError):
        ZeroPlanner(model=DummyDynamicsModel(), **kwargs)


def test_planner_conforms_to_controller_and_tracks_state():
    model = DummyDynamicsModel()
    planner = ZeroPlanner(model=model, action_dim=3, horizon=5, num_samples=16)

    assert isinstance(planner, Controller)

    obs = TensorDict({"render": torch.zeros(3, 8, 8)}, batch_size=[])
    a0 = planner(obs, step=0)
    assert a0.shape == (3,)
    assert model.calls[-1] == (None, None)

    a1 = planner(obs, step=1)
    assert a1.shape == (3,)
    assert model.calls[-1][1] == 1
    assert torch.equal(model.calls[-1][0], a0)

    # Calling with step=0 resets episodic state automatically
    planner(obs, step=0)
    assert model.calls[-1] == (None, None)


def test_planner_works_with_collector():
    env = TorchGymEnv(_DummyBoxEnv(action_dim=2))
    planner = ZeroPlanner(model=DummyDynamicsModel(), action_dim=2, horizon=4, num_samples=8)
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0))

    collector = Collector(env=env, controller=planner, transform=transform, num_episodes=1, max_steps=5, verbose=False)
    ep = collector.collect_one_episode()
    assert ep.action.shape == (5, 2)
