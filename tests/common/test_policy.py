import gymnasium as gym
import pytest
import torch

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

    breakout_pol = BreakoutRandomPolicy(env=_DummyEnv(gym.spaces.Discrete(4)))
    car_pol = CarRacingRandomPolicy(env=_DummyEnv(gym.spaces.Box(-1.0, 1.0, (3,))))
    adroit_pol = AdroitHandRandomPolicy(env=_DummyEnv(gym.spaces.Box(-1.0, 1.0, (28,))))
    franka_pol = FrankaKitchenRandomPolicy(env=_DummyEnv(gym.spaces.Box(-1.0, 1.0, (9,))))

    for pol in (breakout_pol, car_pol, adroit_pol, franka_pol):
        assert isinstance(pol, BasePolicy)
        pol.reset()

    dummy_obs = torch.zeros((3, 64, 64), dtype=torch.float32)
    b_act = breakout_pol(dummy_obs, step=0)
    assert isinstance(b_act, torch.Tensor)
    assert b_act.ndim == 0 and b_act.dtype == torch.long
    assert car_pol(dummy_obs, step=0).shape == (3,)
    assert adroit_pol(dummy_obs, step=0).shape == (28,)
    assert franka_pol(dummy_obs, step=0).shape == (9,)


def test_policy_env_validation():
    # DiscretePolicy rejects Box space
    with pytest.raises(ValueError, match="requires a Discrete action_space"):
        BreakoutRandomPolicy(env=_DummyEnv(gym.spaces.Box(-1.0, 1.0, (3,))))

    # ContinuousPolicy rejects Discrete space
    with pytest.raises(ValueError, match="requires a continuous Box action_space"):
        CarRacingRandomPolicy(env=_DummyEnv(gym.spaces.Discrete(4)))

    # ContinuousPolicy rejects wrong action_dim
    with pytest.raises(ValueError, match="expects action_dim=3"):
        CarRacingRandomPolicy(env=_DummyEnv(gym.spaces.Box(-1.0, 1.0, (5,))))

    # Calling sample_random without setting env raises RuntimeError
    pol = CarRacingRandomPolicy()
    dummy_obs = torch.zeros((3, 64, 64), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="env is not set"):
        pol(dummy_obs, step=0)

