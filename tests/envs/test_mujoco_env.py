import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from hydra.utils import instantiate

from wiskers.envs.adroit_hand.env import AdroitHandEnv
from wiskers.envs.franka_kitchen.env import FrankaKitchenEnv
from wiskers.envs.gym_env import GymEnv
from wiskers.envs.inverted_pendulum.env import InvertedPendulumEnv
from wiskers.envs.mujoco_env import MujocoEnv, MujocoPhysics
from wiskers.envs.shadow_hand.env import ShadowHandEnv
from wiskers.pipeline.config import load_config


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


def test_mujoco_env_default_physics_across_envs():
    env_factories = (
        lambda: InvertedPendulumEnv(),
        lambda: FrankaKitchenEnv(tasks_to_complete=["microwave"]),
        lambda: AdroitHandEnv(task="door"),
        lambda: ShadowHandEnv(task="block"),
    )
    for factory in env_factories:
        with factory() as env:
            assert env.gravity == pytest.approx((0.0, 0.0, -9.81))
            assert env.physics == MujocoPhysics()


def test_mujoco_physics_validation():
    with pytest.raises(ValueError, match="length 3"):
        MujocoPhysics(gravity=(0.0, -9.81))

    for bad_timestep in (0.0, -0.01):
        with pytest.raises(ValueError, match="positive"):
            MujocoPhysics(timestep=bad_timestep)

    with pytest.raises(ValueError, match="non-negative"):
        MujocoPhysics(friction_scale=-0.1)

    for bad_mass in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            MujocoPhysics(mass_scale=bad_mass)

    with pytest.raises(ValueError, match="non-negative"):
        MujocoPhysics(damping_scale=-0.5)

    valid = MujocoPhysics(gravity=[0, 0, -1.62], friction_scale=0.0, damping_scale=0.0)
    assert valid.gravity == (0.0, 0.0, -1.62)
    assert isinstance(valid.gravity, tuple)


def test_mujoco_env_physics_overrides_and_reset_persistence():
    with InvertedPendulumEnv() as nom_env:
        nom_friction = nom_env.model.geom_friction.copy()
        nom_mass = nom_env.model.body_mass.copy()
        nom_inertia = nom_env.model.body_inertia.copy()
        nom_subtreemass = nom_env.model.body_subtreemass.copy()
        nom_damping = nom_env.model.dof_damping.copy()

    override = MujocoPhysics(
        gravity=(1.0, -2.0, -3.5),
        timestep=0.005,
        friction_scale=0.5,
        mass_scale=1.5,
        damping_scale=2.0,
    )
    with InvertedPendulumEnv(physics=override) as env:
        for _ in range(2):
            assert env.gravity == pytest.approx((1.0, -2.0, -3.5))
            assert env.timestep == pytest.approx(0.005)
            np.testing.assert_allclose(env.model.geom_friction, nom_friction * 0.5)
            np.testing.assert_allclose(env.model.body_mass, nom_mass * 1.5)
            np.testing.assert_allclose(env.model.body_inertia, nom_inertia * 1.5)
            np.testing.assert_allclose(env.model.body_subtreemass, nom_subtreemass * 1.5)
            np.testing.assert_allclose(env.model.dof_damping, nom_damping * 2.0)
            env.reset(seed=0)


def test_mujoco_env_apply_physics_is_idempotent_and_restorable():
    with InvertedPendulumEnv() as env:
        nom_gravity = env.gravity
        nom_timestep = env.timestep
        nom_friction = env.model.geom_friction.copy()
        nom_mass = env.model.body_mass.copy()
        nom_inertia = env.model.body_inertia.copy()
        nom_subtreemass = env.model.body_subtreemass.copy()
        nom_damping = env.model.dof_damping.copy()

        scaled = MujocoPhysics(mass_scale=2.0, friction_scale=0.5, damping_scale=1.5)
        env.apply_physics(scaled)
        env.apply_physics(scaled)
        np.testing.assert_allclose(env.model.geom_friction, nom_friction * 0.5)
        np.testing.assert_allclose(env.model.body_mass, nom_mass * 2.0)
        np.testing.assert_allclose(env.model.body_inertia, nom_inertia * 2.0)
        np.testing.assert_allclose(env.model.body_subtreemass, nom_subtreemass * 2.0)
        np.testing.assert_allclose(env.model.dof_damping, nom_damping * 1.5)

        env.apply_physics(MujocoPhysics())
        assert env.gravity == pytest.approx(nom_gravity)
        assert env.timestep == pytest.approx(nom_timestep)
        np.testing.assert_allclose(env.model.geom_friction, nom_friction)
        np.testing.assert_allclose(env.model.body_mass, nom_mass)
        np.testing.assert_allclose(env.model.body_inertia, nom_inertia)
        np.testing.assert_allclose(env.model.body_subtreemass, nom_subtreemass)
        np.testing.assert_allclose(env.model.dof_damping, nom_damping)


def test_mujoco_env_physics_affects_simulation():
    zero_action = torch.zeros(1, dtype=torch.float32)

    with InvertedPendulumEnv(physics=MujocoPhysics(gravity=(0.0, 0.0, 0.0))) as zero_g_env:
        obs_zero, _ = zero_g_env.reset(seed=0)
        for _ in range(5):
            obs_zero, _, _, _, _ = zero_g_env.step(zero_action)

    with InvertedPendulumEnv(physics=MujocoPhysics(gravity=(0.0, 0.0, -25.0))) as high_g_env:
        obs_high, _ = high_g_env.reset(seed=0)
        for _ in range(5):
            obs_high, _, _, _, _ = high_g_env.step(zero_action)

    assert abs(float(obs_high["state:qpos"][1].item())) > abs(float(obs_zero["state:qpos"][1].item()))

    push_action = torch.tensor([1.0], dtype=torch.float32)
    with InvertedPendulumEnv(physics=MujocoPhysics(damping_scale=0.0)) as no_damp_env:
        obs_no_damp, _ = no_damp_env.reset(seed=0)
        for _ in range(5):
            obs_no_damp, _, _, _, _ = no_damp_env.step(push_action)

    with InvertedPendulumEnv(physics=MujocoPhysics(damping_scale=10.0)) as high_damp_env:
        obs_high_damp, _ = high_damp_env.reset(seed=0)
        for _ in range(5):
            obs_high_damp, _, _, _, _ = high_damp_env.step(push_action)

    assert not torch.allclose(obs_no_damp["state:qvel"], obs_high_damp["state:qvel"])


def test_mujoco_env_render_frame_none_raises(monkeypatch):
    monkeypatch.setattr(MujocoRenderer, "render", lambda self, *args, **kwargs: None)
    with InvertedPendulumEnv() as env:
        with pytest.raises(RuntimeError, match="render\\(\\) returned None"):
            env.reset(seed=0)


def test_mujoco_env_hydra_physics_instantiation_and_cli_overrides():
    cfg = load_config(
        "dataset/inverted_pendulum",
        overrides=[
            "data_module.collector.env.physics.gravity=[0.0,0.0,-3.71]",
            "data_module.collector.env.physics.timestep=0.01",
            "data_module.collector.env.physics.mass_scale=1.5",
        ],
    )
    with instantiate(cfg.data_module.collector.env, _convert_="all") as env:
        assert env.gravity == pytest.approx((0.0, 0.0, -3.71))
        assert env.timestep == pytest.approx(0.01)
        assert env.physics.mass_scale == pytest.approx(1.5)


def test_mujoco_env_obs_fields_transforms_selected_fields():
    with pytest.raises(KeyError, match="unknown"):
        InvertedPendulumEnv(obs_fields={"unknown": None})
    with pytest.raises(KeyError, match="state:obj_qvel"):
        ShadowHandEnv(task="reach", obs_fields={"state:obj_qvel": None})

    with InvertedPendulumEnv() as nom_env:
        nom_obs, _ = nom_env.reset(seed=0)
        nom_step_obs, _, _, _, _ = nom_env.step(torch.zeros(1))

    with InvertedPendulumEnv(obs_fields={"state:qpos": None, "state:qvel": lambda t: t * 2.0}) as env:
        assert set(env._obs_transforms.keys()) == {"state:qvel"}
        obs, _ = env.reset(seed=0)
        assert set(obs.keys()) == {"render", "state:qpos", "state:qvel"}
        assert torch.allclose(obs["state:qpos"], nom_obs["state:qpos"])
        assert torch.allclose(obs["state:qvel"], nom_obs["state:qvel"] * 2.0)

        step_obs, _, _, _, _ = env.step(torch.zeros(1))
        assert set(step_obs.keys()) == {"render", "state:qpos", "state:qvel"}
        assert torch.allclose(step_obs["state:qpos"], nom_step_obs["state:qpos"])
        assert torch.allclose(step_obs["state:qvel"], nom_step_obs["state:qvel"] * 2.0)


def test_all_dataset_configs_obs_fields_instantiation():
    for name in (
        "adroit_hand",
        "breakout",
        "car_racing",
        "franka_kitchen",
        "inverted_pendulum",
        "shadow_hand",
    ):
        cfg = load_config(f"dataset/{name}")
        with instantiate(cfg.data_module.collector.env, _convert_="all") as env:
            obs, _ = env.reset(seed=0)
            assert set(obs.keys()) == set(env.obs_keys)
            assert set(obs.keys()) == set(cfg.data_module.collector.env.obs_fields.keys())
            assert obs["render"].shape == (64, 64, 3)
            assert obs["render"].dtype == torch.uint8
