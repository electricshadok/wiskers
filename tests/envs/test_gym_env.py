import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.gym_env import GymEnv
from wiskers.envs.utils import ResizeUint8, patch_mujoco_joint_eq


class DummyGymEnv(gym.Env):
    def __init__(self, shape=(120, 160, 3)):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 100, {}

    def step(self, action):
        return np.ones(self.observation_space.shape, dtype=np.uint8) * 200, 1.0, False, False, {}


def test_gym_env_reset_and_step():
    gym_env = GymEnv(DummyGymEnv(shape=(120, 160, 3)))

    obs, info = gym_env.reset(seed=0)
    assert isinstance(obs, TensorDict)
    assert obs.batch_size == torch.Size([])
    assert list(obs.keys()) == []

    action = torch.tensor([0.5, -0.5], dtype=torch.float32)
    next_obs, reward, term, trunc, info = gym_env.step(action)
    assert isinstance(next_obs, TensorDict)
    assert next_obs.batch_size == torch.Size([])
    assert list(next_obs.keys()) == []
    assert reward == 1.0


def test_gym_env_builds_action_spec_from_gym_spaces():
    box_env = GymEnv(DummyGymEnv())
    assert isinstance(box_env.action_spec, BoundedActionSpec)
    assert box_env.action_spec.shape == (2,)
    assert box_env.action_spec.dtype == torch.float32
    assert box_env.action_spec.contains(box_env.action_spec.sample())

    class IntBoxGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.int64)

    with pytest.raises(ValueError, match="Unsupported Box action dtype"):
        GymEnv(IntBoxGymEnv())

    class DiscreteGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Discrete(4)

    disc_env = GymEnv(DiscreteGymEnv())
    assert isinstance(disc_env.action_spec, DiscreteActionSpec)
    assert disc_env.action_spec.n == 4
    assert disc_env.action_spec.contains(disc_env.action_spec.sample())

    class UnsupportedGymEnv(DummyGymEnv):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.MultiDiscrete([2, 3])

    with pytest.raises(ValueError, match="Unsupported action space"):
        GymEnv(UnsupportedGymEnv())


def test_patch_mujoco_joint_eq():
    import mujoco

    patch_mujoco_joint_eq()
    assert mujoco.mjtJoint.mjJNT_FREE == int(mujoco.mjtJoint.mjJNT_FREE)
    assert mujoco.mjtJoint.mjJNT_FREE == np.int32(int(mujoco.mjtJoint.mjJNT_FREE))


def test_gym_env_context_manager():
    closed = False

    class TrackingEnv(DummyGymEnv):
        def close(self):
            nonlocal closed
            closed = True

    with GymEnv(TrackingEnv()) as wrapped:
        assert isinstance(wrapped, GymEnv)
        assert not closed
    assert closed


def test_gym_env_obs_fields_validation():
    empty_env = GymEnv(DummyGymEnv(), obs_fields={})
    assert empty_env._obs_transforms == {}

    with pytest.raises(KeyError, match="unknown"):
        GymEnv(DummyGymEnv(), obs_fields={"unknown": None})


def test_resize_uint8():
    transform = ResizeUint8((64, 64))

    resized_3d = transform(torch.full((120, 160, 3), 150.0, dtype=torch.float32))
    assert resized_3d.shape == (64, 64, 3)
    assert resized_3d.dtype == torch.uint8
    assert torch.all(resized_3d == 150)

    resized_2d = transform(torch.full((120, 160), 200, dtype=torch.uint8))
    assert resized_2d.shape == (64, 64)
    assert resized_2d.dtype == torch.uint8

    same_size = transform(torch.full((64, 64, 3), 42.0, dtype=torch.float32))
    assert same_size.shape == (64, 64, 3)
    assert same_size.dtype == torch.uint8
    assert torch.all(same_size == 42)

    for bad_size in ((64,), (0, 64), (-1, 64)):
        with pytest.raises(ValueError, match="2-element sequence of positive ints"):
            ResizeUint8(bad_size)
