import gymnasium as gym
import pytest
import torch

from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs import TorchGymEnv
from wiskers.envs.adroit_hand.policies import AdroitHandPolicy, AdroitHandRandomPolicy
from wiskers.envs.breakout.policies import BreakoutPolicy, BreakoutRandomPolicy
from wiskers.envs.car_racing.policies import CarRacingPolicy, CarRacingRandomPolicy
from wiskers.envs.franka.policies import FrankaKitchenPolicy, FrankaKitchenRandomPolicy
from wiskers.policies import ContinuousPolicy, DiscretePolicy, Policy, RandomPolicy


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
        IncompletePolicy()(torch.zeros(3, 64, 64))


def test_random_policy_discrete():
    pol = RandomPolicy(gym.spaces.Discrete(4))
    assert pol.action_dim == 1
    act = pol(torch.zeros(3, 64, 64))
    assert isinstance(act, torch.Tensor)
    assert act.shape == (1,) and act.dtype == torch.int64
    assert 0 <= act.item() < 4


def test_random_policy_box():
    pol = RandomPolicy(gym.spaces.Box(-1.0, 1.0, shape=(5,)))
    assert pol.action_dim == 5
    act = pol(torch.zeros(3, 64, 64))
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

    dummy_obs = torch.zeros((3, 64, 64), dtype=torch.float32)
    b_act = breakout_pol(dummy_obs, step=0)
    assert isinstance(b_act, torch.Tensor)
    assert b_act.shape == (1,) and b_act.dtype == torch.long
    assert car_pol(dummy_obs, step=0).shape == (3,)
    assert adroit_pol(dummy_obs, step=0).shape == (28,)
    assert franka_pol(dummy_obs, step=0).shape == (9,)


def test_collector_policy_env_validation():
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    # Discrete env with Continuous policy -> ValueError
    discrete_env = TorchGymEnv(_DummyEnv(gym.spaces.Discrete(4)), image_size=(64, 64))
    box_env = TorchGymEnv(_DummyEnv(gym.spaces.Box(-1.0, 1.0, shape=(3,))), image_size=(64, 64))

    with pytest.raises(ValueError, match="Env has Discrete action space"):
        Collector(env=discrete_env, policy=CarRacingRandomPolicy(), transform=transform)

    # Box env with wrong action_dim -> ValueError
    with pytest.raises(ValueError, match="Env has Box action space with shape"):
        Collector(env=box_env, policy=FrankaKitchenRandomPolicy(), transform=transform)
