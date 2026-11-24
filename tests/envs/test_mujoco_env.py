import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wiskers.envs.adroit_hand.env import AdroitHandEnv
from wiskers.envs.franka_kitchen.env import FrankaKitchenEnv
from wiskers.envs.gym_env import GymEnv
from wiskers.envs.inverted_pendulum.env import InvertedPendulumEnv
from wiskers.envs.mujoco_env import MujocoEnv
from wiskers.envs.shadow_hand.env import ShadowHandEnv


@pytest.fixture(autouse=True)
def mock_mujoco_renderer(monkeypatch):
    """Bypass OpenGL/GLFW window creation during unit tests while keeping real MuJoCo physics."""
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )


def test_mujoco_envs_inherit_from_mujoco_env():
    for env_cls in (InvertedPendulumEnv, FrankaKitchenEnv, AdroitHandEnv, ShadowHandEnv):
        assert issubclass(env_cls, MujocoEnv)
        assert issubclass(env_cls, GymEnv)


def test_mujoco_env_default_gravity_across_envs():
    env_factories = (
        lambda: InvertedPendulumEnv(image_size=(64, 64)),
        lambda: FrankaKitchenEnv(image_size=(64, 64), tasks_to_complete=["microwave"]),
        lambda: AdroitHandEnv(task="door", image_size=(64, 64)),
        lambda: ShadowHandEnv(task="block", image_size=(64, 64)),
    )
    for factory in env_factories:
        with factory() as env:
            assert env.gravity == pytest.approx((0.0, 0.0, -9.81))


def test_mujoco_env_gravity_persists_across_reset():
    with InvertedPendulumEnv(image_size=(64, 64), gravity=(0.0, 0.0, -1.62)) as moon_env:
        assert moon_env.gravity == pytest.approx((0.0, 0.0, -1.62))
        moon_env.reset(seed=0)
        assert moon_env.gravity == pytest.approx((0.0, 0.0, -1.62))

    with InvertedPendulumEnv(image_size=(64, 64), gravity=(1.0, -2.0, -3.5)) as vec_env:
        assert vec_env.gravity == pytest.approx((1.0, -2.0, -3.5))
        vec_env.reset(seed=0)
        assert vec_env.gravity == pytest.approx((1.0, -2.0, -3.5))


def test_mujoco_env_gravity_affects_simulation_physics():
    zero_action = torch.zeros(1, dtype=torch.float32)

    with InvertedPendulumEnv(image_size=(64, 64), gravity=(0.0, 0.0, 0.0)) as zero_g_env:
        obs_zero, _ = zero_g_env.reset(seed=0)
        for _ in range(5):
            obs_zero, _, _, _, _ = zero_g_env.step(zero_action)

    with InvertedPendulumEnv(image_size=(64, 64), gravity=(0.0, 0.0, -25.0)) as high_g_env:
        obs_high, _ = high_g_env.reset(seed=0)
        for _ in range(5):
            obs_high, _, _, _, _ = high_g_env.step(zero_action)

    assert abs(float(obs_high["state:qpos"][1].item())) > abs(float(obs_zero["state:qpos"][1].item()))


def test_mujoco_env_invalid_gravity_raises():
    with pytest.raises(ValueError, match="length 3"):
        InvertedPendulumEnv(gravity=(0.0, -9.81))


def test_mujoco_env_render_frame_none_raises(monkeypatch):
    monkeypatch.setattr(MujocoRenderer, "render", lambda self, *args, **kwargs: None)
    with InvertedPendulumEnv(image_size=(64, 64)) as env:
        with pytest.raises(RuntimeError, match="render\\(\\) returned None"):
            env.reset(seed=0)


def test_mujoco_env_hydra_gravity_instantiation():
    vec_cfg = OmegaConf.create(
        {
            "_target_": "wiskers.envs.inverted_pendulum.env.InvertedPendulumEnv",
            "image_size": [64, 64],
            "gravity": [0.0, 0.0, -3.71],
        }
    )
    with instantiate(vec_cfg, _convert_="all") as vec_env:
        assert vec_env.gravity == pytest.approx((0.0, 0.0, -3.71))
