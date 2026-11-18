import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from wiskers.controller import Controller
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.adroit_hand.policies import AdroitHandPolicy, AdroitHandRandomPolicy
from wiskers.envs.breakout.policies import BreakoutPolicy, BreakoutRandomPolicy
from wiskers.envs.car_racing.policies import CarRacingPolicy, CarRacingRandomPolicy
from wiskers.envs.franka_kitchen.policies import FrankaKitchenPolicy, FrankaKitchenRandomPolicy
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.policies.base import ContinuousPolicy, DiscretePolicy, Policy, RandomPolicy


class _DummyEnv(gym.Env):
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=int)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 1.0, False, False, {}


class _ConcretePolicy(Policy):
    action_dim = 2

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        return torch.zeros(self.action_dim, dtype=torch.float32)


def test_policy_interface():
    pol = _ConcretePolicy()
    assert pol.action_dim == 2
    pol.reset()
    act = pol(torch.zeros(3, 64, 64))
    assert act.shape == (2,)

    class IncompletePolicy(Policy):
        action_dim = 1

    with pytest.raises(NotImplementedError):
        IncompletePolicy()(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))


def test_random_policy_discrete():
    pol = RandomPolicy(gym.spaces.Discrete(4))
    assert pol.action_dim == 1
    act = pol(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))
    assert isinstance(act, torch.Tensor)
    assert act.shape == (1,) and act.dtype == torch.int64
    assert 0 <= act.item() < 4


def test_random_policy_box():
    pol = RandomPolicy(gym.spaces.Box(-1.0, 1.0, shape=(5,)))
    assert pol.action_dim == 5
    act = pol(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))
    assert isinstance(act, torch.Tensor)
    assert act.shape == (5,)
    assert act.dtype == torch.float32


def test_env_policies_inherit_from_policy():
    assert issubclass(BreakoutPolicy, Policy)
    assert issubclass(BreakoutPolicy, DiscretePolicy)
    assert issubclass(CarRacingPolicy, Policy)
    assert issubclass(CarRacingPolicy, ContinuousPolicy)
    assert issubclass(AdroitHandPolicy, Policy)
    assert issubclass(AdroitHandPolicy, ContinuousPolicy)
    assert issubclass(FrankaKitchenPolicy, Policy)
    assert issubclass(FrankaKitchenPolicy, ContinuousPolicy)

    breakout_pol = BreakoutRandomPolicy()
    car_pol = CarRacingRandomPolicy()
    adroit_pol = AdroitHandRandomPolicy()
    franka_pol = FrankaKitchenRandomPolicy()

    for pol in (breakout_pol, car_pol, adroit_pol, franka_pol):
        assert isinstance(pol, Policy)
        pol.reset()

    dummy_obs = TensorDict({"render": torch.zeros((3, 64, 64), dtype=torch.float32)}, batch_size=[])
    b_act = breakout_pol(dummy_obs, step=0)
    assert isinstance(b_act, torch.Tensor)
    assert b_act.shape == (1,) and b_act.dtype == torch.long
    assert car_pol(dummy_obs, step=0).shape == (3,)
    assert adroit_pol(dummy_obs, step=0).shape == (28,)
    assert franka_pol(dummy_obs, step=0).shape == (9,)


def test_collector_policy_env_validation():
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    # Discrete env with Continuous policy -> ValueError
    discrete_env = TorchGymEnv(_DummyEnv(gym.spaces.Discrete(4)))
    box_env = TorchGymEnv(_DummyEnv(gym.spaces.Box(-1.0, 1.0, shape=(3,))))

    with pytest.raises(ValueError, match="Env has Discrete action space"):
        Collector(env=discrete_env, controller=CarRacingRandomPolicy(), transform=transform)

    # Box env with wrong action_dim -> ValueError
    with pytest.raises(ValueError, match="Env has Box action space with shape"):
        Collector(env=box_env, controller=FrankaKitchenRandomPolicy(), transform=transform)


def test_controller_protocol_supports_policy_and_custom_planner():
    class DummyPlanner:
        action_dim: int = 3

        def __init__(self) -> None:
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def __call__(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
            return torch.zeros(self.action_dim, dtype=torch.float32)

    planner = DummyPlanner()
    assert isinstance(planner, Controller)
    assert isinstance(CarRacingRandomPolicy(), Controller)

    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    env = TorchGymEnv(_DummyEnv(gym.spaces.Box(-1.0, 1.0, shape=(3,))))
    collector = Collector(env=env, controller=planner, transform=transform, num_episodes=1, max_steps=3, verbose=False)
    ep = collector.collect_one_episode()

    assert planner.resets == 1
    assert ep.action.shape == (3, 3)
