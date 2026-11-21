import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.controller import Controller
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.planner.random_shooting import RandomShootingPlanner


class TargetDynamicsModel:
    """Toy dynamics model whose imagined return peaks at a target action."""

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target

    def encode_step(
        self,
        obs: TensorDict,
        prev_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return obs["render"]

    def imagine_returns(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return -((actions - self.target) ** 2).sum(dim=(-1, -2))


class _DummyBoxEnv(gym.Env):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((8, 8, 3), dtype=np.uint8), 1.0, False, False, {}


def test_random_shooting_conforms_to_controller():
    spec = BoundedActionSpec(low=torch.full((2,), -1.0), high=torch.full((2,), 1.0))
    planner = RandomShootingPlanner(
        model=TargetDynamicsModel(torch.zeros(2)),
        action_spec=spec,
        horizon=4,
        num_samples=32,
    )
    assert isinstance(planner, Controller)
    assert planner.action_spec == spec


def test_random_shooting_action_shape_dtype_and_bounds():
    low = torch.tensor([-2.0, 0.0, -0.5], dtype=torch.float32)
    high = torch.tensor([0.5, 1.0, 0.25], dtype=torch.float32)
    spec = BoundedActionSpec(low=low, high=high)
    planner = RandomShootingPlanner(
        model=TargetDynamicsModel(torch.tensor([-1.0, 0.5, 0.0])),
        action_spec=spec,
        horizon=5,
        num_samples=64,
    )
    obs = TensorDict({"render": torch.zeros(3, 8, 8)}, batch_size=[])
    action = planner(obs, step=0)

    assert action.shape == (3,)
    assert action.dtype == torch.float32
    assert spec.contains(action)


def test_random_shooting_selects_high_return_action():
    torch.manual_seed(0)
    target = torch.tensor([0.8, -0.7])
    spec = BoundedActionSpec(low=torch.full((2,), -1.0), high=torch.full((2,), 1.0))
    planner = RandomShootingPlanner(
        model=TargetDynamicsModel(target),
        action_spec=spec,
        horizon=3,
        num_samples=512,
    )
    obs = TensorDict({"render": torch.zeros(3, 8, 8)}, batch_size=[])
    action = planner(obs, step=0)

    assert torch.linalg.norm(action - target) < 0.35


def test_random_shooting_invalid_action_spec():
    model = TargetDynamicsModel(torch.zeros(2))
    with pytest.raises(TypeError, match="BoundedActionSpec"):
        RandomShootingPlanner(model=model, action_spec=DiscreteActionSpec(n=4))  # type: ignore[arg-type]

    int_spec = BoundedActionSpec(low=torch.tensor([0, 0]), high=torch.tensor([2, 2]))
    with pytest.raises(ValueError, match="floating-point"):
        RandomShootingPlanner(model=model, action_spec=int_spec)

    inf_spec = BoundedActionSpec(low=torch.tensor([-float("inf"), -1.0]), high=torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        RandomShootingPlanner(model=model, action_spec=inf_spec)


def test_random_shooting_collector_integration():
    torch.manual_seed(0)
    env = TorchGymEnv(_DummyBoxEnv(action_dim=2))
    assert isinstance(env.action_spec, BoundedActionSpec)
    planner = RandomShootingPlanner(
        model=TargetDynamicsModel(torch.tensor([0.5, -0.5])),
        action_spec=env.action_spec,
        horizon=4,
        num_samples=64,
    )
    transform = EpisodeToTensor(image_size=(3, 8, 8), obs_range=(0.0, 1.0))

    collector = Collector(env=env, controller=planner, transform=transform, num_episodes=1, max_steps=6, verbose=False)
    ep = collector.collect_one_episode()

    assert ep.action.shape == (6, 2)
    assert ep.action.dtype == torch.float32
    assert torch.all(ep.action >= -1.0) and torch.all(ep.action <= 1.0)
