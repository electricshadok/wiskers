import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.controller import Controller, validate_controller
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.adroit_hand.policies import AdroitHandPolicy, AdroitHandRandomPolicy
from wiskers.envs.breakout.policies import BreakoutPolicy, BreakoutRandomPolicy
from wiskers.envs.car_racing.policies import CarRacingPolicy, CarRacingRandomPolicy
from wiskers.envs.franka_kitchen.policies import FrankaKitchenPolicy, FrankaKitchenRandomPolicy
from wiskers.envs.shadow_hand.policies import ShadowHandPolicy, ShadowHandRandomPolicy
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.policy.base import ContinuousPolicy, DiscretePolicy, Policy, RandomPolicy


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
    def forward(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
        return torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)


def test_policy_interface():
    spec = BoundedActionSpec(low=torch.full((2,), -1.0), high=torch.full((2,), 1.0))
    pol = _ConcretePolicy(spec)
    assert pol.action_spec == spec
    assert pol.action_spec.action_dim == 2
    pol.reset()
    act = pol(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))
    assert act.shape == (2,)

    class IncompletePolicy(Policy):
        pass

    with pytest.raises(NotImplementedError):
        IncompletePolicy(DiscreteActionSpec(n=4))(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))

    with pytest.raises(TypeError, match="DiscretePolicy requires DiscreteActionSpec"):
        DiscretePolicy(spec)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ContinuousPolicy requires BoundedActionSpec"):
        ContinuousPolicy(DiscreteActionSpec(n=4))  # type: ignore[arg-type]


def test_random_policy_discrete():
    pol = RandomPolicy(DiscreteActionSpec(n=4))
    assert pol.action_spec.action_dim == 1
    act = pol(TensorDict({"render": torch.zeros(3, 64, 64)}, batch_size=[]))
    assert isinstance(act, torch.Tensor)
    assert act.shape == (1,) and act.dtype == torch.int64
    assert 0 <= act.item() < 4


def test_random_policy_bounded():
    pol = RandomPolicy(
        BoundedActionSpec(
            low=torch.full((5,), -1.0),
            high=torch.full((5,), 1.0),
        )
    )
    assert pol.action_spec.action_dim == 5
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
    assert issubclass(ShadowHandPolicy, Policy)
    assert issubclass(ShadowHandPolicy, ContinuousPolicy)

    breakout_pol = BreakoutRandomPolicy()
    car_pol = CarRacingRandomPolicy()
    adroit_pol = AdroitHandRandomPolicy()
    franka_pol = FrankaKitchenRandomPolicy()
    shadow_pol = ShadowHandRandomPolicy()

    for pol in (breakout_pol, car_pol, adroit_pol, franka_pol, shadow_pol):
        assert isinstance(pol, Policy)
        pol.reset()

    dummy_obs = TensorDict({"render": torch.zeros((3, 64, 64), dtype=torch.float32)}, batch_size=[])
    b_act = breakout_pol(dummy_obs, step=0)
    assert isinstance(b_act, torch.Tensor)
    assert b_act.shape == (1,) and b_act.dtype == torch.long
    assert car_pol(dummy_obs, step=0).shape == (3,)
    assert adroit_pol(dummy_obs, step=0).shape == (28,)
    assert franka_pol(dummy_obs, step=0).shape == (9,)
    assert shadow_pol(dummy_obs, step=0).shape == (20,)


def test_collector_policy_env_validation():
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))

    discrete_env = TorchGymEnv(_DummyEnv(gym.spaces.Discrete(4)))
    discrete_5_env = TorchGymEnv(_DummyEnv(gym.spaces.Discrete(5)))
    car_racing_box_env = TorchGymEnv(
        _DummyEnv(
            gym.spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        )
    )
    symmetric_box_env = TorchGymEnv(_DummyEnv(gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)))

    # Discrete env with Continuous policy -> ValueError
    with pytest.raises(ValueError, match="does not match environment action_spec"):
        Collector(env=discrete_env, controller=CarRacingRandomPolicy(), transform=transform)

    # Discrete env with wrong discrete cardinality -> ValueError
    with pytest.raises(ValueError, match="does not match environment action_spec"):
        Collector(env=discrete_5_env, controller=BreakoutRandomPolicy(), transform=transform)

    # Bounded env with wrong action_dim -> ValueError
    with pytest.raises(ValueError, match="does not match environment action_spec"):
        Collector(env=car_racing_box_env, controller=FrankaKitchenRandomPolicy(), transform=transform)

    # Bounded env with matching dim (3,) but mismatched bounds -> ValueError
    with pytest.raises(ValueError, match="does not match environment action_spec"):
        Collector(env=symmetric_box_env, controller=CarRacingRandomPolicy(), transform=transform)

    # Direct validate_controller calls succeed when specs match exactly
    validate_controller(BreakoutRandomPolicy(), discrete_env.action_spec)
    validate_controller(CarRacingRandomPolicy(), car_racing_box_env.action_spec)


def test_controller_protocol_supports_policy_and_custom_planner():
    class DummyPlanner:
        def __init__(self, action_spec: BoundedActionSpec) -> None:
            self.action_spec = action_spec
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

        def __call__(self, obs: TensorDict, step: int = 0) -> torch.Tensor:
            return torch.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)

    spec = BoundedActionSpec(low=torch.full((3,), -1.0), high=torch.full((3,), 1.0))
    planner = DummyPlanner(spec)
    assert isinstance(planner, Controller)
    assert isinstance(CarRacingRandomPolicy(), Controller)

    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    env = TorchGymEnv(_DummyEnv(gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)))
    collector = Collector(env=env, controller=planner, transform=transform, num_episodes=1, max_steps=3, verbose=False)
    ep = collector.collect_one_episode()

    assert planner.resets == 1
    assert ep.action.shape == (3, 3)
