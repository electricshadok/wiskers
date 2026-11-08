import gymnasium as gym
import numpy as np
import pytest

from wiskers.common.policy import BasePolicy
from wiskers.envs.adroit_hand.policies import AdroitHandPolicy, AdroitHandRandomPolicy
from wiskers.envs.breakout.policies import BreakoutPolicy, BreakoutRandomPolicy
from wiskers.envs.car_racing.policies import CarRacingPolicy, CarRacingRandomPolicy
from wiskers.envs.franka.policies import FrankaKitchenPolicy, FrankaKitchenRandomPolicy


class _DummyEnv(gym.Env):
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space


def test_base_policy_is_abstract():
    with pytest.raises(TypeError):
        BasePolicy()


def test_env_policies_inherit_from_base_policy():
    assert issubclass(BreakoutPolicy, BasePolicy)
    assert issubclass(CarRacingPolicy, BasePolicy)
    assert issubclass(AdroitHandPolicy, BasePolicy)
    assert issubclass(FrankaKitchenPolicy, BasePolicy)

    breakout_pol = BreakoutRandomPolicy()
    car_pol = CarRacingRandomPolicy()
    adroit_pol = AdroitHandRandomPolicy()
    franka_pol = FrankaKitchenRandomPolicy()

    for pol in (breakout_pol, car_pol, adroit_pol, franka_pol):
        assert isinstance(pol, BasePolicy)
        pol.reset()

    dummy_obs = np.zeros((64, 64, 3), dtype=np.uint8)
    assert isinstance(breakout_pol(dummy_obs, _DummyEnv(gym.spaces.Discrete(4)), step=0), int)
    assert car_pol(dummy_obs, _DummyEnv(gym.spaces.Box(-1.0, 1.0, (3,))), step=0).shape == (3,)
    assert adroit_pol(dummy_obs, _DummyEnv(gym.spaces.Box(-1.0, 1.0, (28,))), step=0).shape == (28,)
    assert franka_pol(dummy_obs, _DummyEnv(gym.spaces.Box(-1.0, 1.0, (9,))), step=0).shape == (9,)
