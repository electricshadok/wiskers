import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from tensordict import TensorDict

from wiskers.data.buffer import ReplayBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.shadow_hand.env import AVAILABLE_TASKS, ShadowHandEnv
from wiskers.envs.shadow_hand.policies import (
    ShadowHandHeuristicPolicy,
    ShadowHandNeuralPolicy,
    ShadowHandPolicy,
    ShadowHandRandomPolicy,
)
from wiskers.pipeline.config import load_config


@pytest.fixture(autouse=True)
def mock_mujoco_renderer(monkeypatch):
    """Bypass OpenGL/GLFW window creation during unit tests while keeping real MuJoCo physics."""
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )


@pytest.fixture
def shadow_env():
    env = ShadowHandEnv(task="block", image_size=(64, 64))
    yield env
    env.close()


def test_shadow_hand_env_observation_shape():
    env = ShadowHandEnv(task="block")
    try:
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, TensorDict)
        assert obs.batch_size == torch.Size([])
        assert obs["render"].shape == (480, 480, 3)
        assert obs["render"].dtype == torch.uint8
        assert obs["state:hand_qpos"].shape == (24,)
        assert obs["state:hand_qpos"].dtype == torch.float32
        assert obs["state:hand_qvel"].shape == (24,)
        assert obs["state:hand_qvel"].dtype == torch.float32
        assert obs["state:obj_qvel"].shape == (6,)
        assert obs["state:obj_qvel"].dtype == torch.float32
        assert obs["state:achieved_goal"].shape == (7,)
        assert obs["state:achieved_goal"].dtype == torch.float32
        assert obs["state:desired_goal"].shape == (7,)
        assert obs["state:desired_goal"].dtype == torch.float32

        action = env.sample_action()
        assert action.shape == (20,)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        assert isinstance(next_obs, TensorDict)
        assert next_obs.batch_size == torch.Size([])
        assert next_obs["render"].shape == (480, 480, 3)
        assert next_obs["state:hand_qpos"].shape == (24,)
        assert next_obs["state:hand_qvel"].shape == (24,)
        assert next_obs["state:obj_qvel"].shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


@pytest.mark.parametrize(
    "task,goal_dim,has_obj_qvel",
    [
        ("reach", 15, False),
        ("block", 7, True),
        ("egg", 7, True),
        ("pen", 7, True),
    ],
)
def test_shadow_hand_state_dimensions_match_farama_spec(
    task: str,
    goal_dim: int,
    has_obj_qvel: bool,
):
    env = ShadowHandEnv(task=task, image_size=(64, 64))
    try:
        raw_obs, _ = env._gym_env.reset(seed=42)
        expected_obs = torch.from_numpy(np.ascontiguousarray(raw_obs["observation"], dtype=np.float32))
        expected_achieved = torch.from_numpy(np.ascontiguousarray(raw_obs["achieved_goal"], dtype=np.float32))
        expected_desired = torch.from_numpy(np.ascontiguousarray(raw_obs["desired_goal"], dtype=np.float32))

        obs, _ = env.reset(seed=42)
        assert obs["state:hand_qpos"].shape == (24,)
        assert obs["state:hand_qvel"].shape == (24,)
        assert obs["state:achieved_goal"].shape == (goal_dim,)
        assert obs["state:desired_goal"].shape == (goal_dim,)
        assert torch.equal(obs["state:hand_qpos"], expected_obs[0:24])
        assert torch.equal(obs["state:hand_qvel"], expected_obs[24:48])
        assert torch.equal(obs["state:achieved_goal"], expected_achieved)
        assert torch.equal(obs["state:desired_goal"], expected_desired)

        if has_obj_qvel:
            assert "state:obj_qvel" in obs.keys()
            assert obs["state:obj_qvel"].shape == (6,)
            assert torch.equal(obs["state:obj_qvel"], expected_obs[48:54])
        else:
            assert "state:obj_qvel" not in obs.keys()

        assert env.sample_action().shape == (20,)
    finally:
        env.close()


def test_shadow_hand_env_invalid_task():
    with pytest.raises(ValueError, match="Unknown task"):
        ShadowHandEnv(task="non_existent_task")


def test_shadow_hand_available_tasks():
    assert set(AVAILABLE_TASKS.keys()) == {"reach", "block", "egg", "pen"}


def test_shadow_random_policy(shadow_env):
    policy = ShadowHandRandomPolicy()
    obs, _ = shadow_env.reset(seed=42)
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (20,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_shadow_heuristic_policy(shadow_env):
    policy = ShadowHandHeuristicPolicy(momentum=0.8, noise_scale=0.2)
    obs, _ = shadow_env.reset(seed=42)
    act1 = policy(obs, step=0)
    act2 = policy(obs, step=1)
    assert act1.shape == (20,)
    assert act2.shape == (20,)
    assert torch.all(act1 >= -1.0) and torch.all(act1 <= 1.0)
    assert torch.all(act2 >= -1.0) and torch.all(act2 <= 1.0)

    policy.reset()
    assert torch.all(policy.current_action == 0.0)


def test_shadow_hand_collector_integration(shadow_env):
    policy = ShadowHandHeuristicPolicy()
    transform = EpisodeToTensor(image_size=(3, 64, 64), obs_range=(0.0, 1.0))
    collector = Collector(
        env=shadow_env,
        controller=policy,
        transform=transform,
        num_episodes=1,
        min_steps=5,
        max_steps=10,
        verbose=False,
    )
    buffer = ReplayBuffer(capacity_episodes=5)
    collected_steps = collector.collect(buffer)

    assert collected_steps == 10
    assert buffer.num_episodes == 1
    assert buffer.num_steps == 10
    assert buffer[0].obs["render"].shape == (10, 64, 64, 3)
    assert buffer[0].obs["render"].dtype == torch.uint8
    assert buffer[0].obs["state:hand_qpos"].shape == (10, 24)
    assert buffer[0].obs["state:hand_qvel"].shape == (10, 24)
    assert buffer[0].obs["state:obj_qvel"].shape == (10, 6)
    assert buffer[0].obs["state:achieved_goal"].shape == (10, 7)
    assert buffer[0].obs["state:desired_goal"].shape == (10, 7)
    assert buffer[0].action.shape == (10, 20)
    assert buffer[0].action.dtype == torch.float32


def test_shadow_neural_policy():
    image_size = (3, 64, 64)
    policy = ShadowHandNeuralPolicy(image_size=image_size, hidden_size=32, num_layers=2)
    assert isinstance(policy, ShadowHandPolicy)

    params = list(policy.parameters())
    assert params and all(p.requires_grad for p in params)

    obs = TensorDict({"render": torch.rand(image_size)}, batch_size=[])
    action = policy(obs, step=0)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (20,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    feat = obs["render"].flatten()
    out = policy.actor(feat)[0]
    out.sum().backward()
    assert all(p.grad is not None for p in params)

    policy.eval()
    assert torch.equal(policy(obs), policy(obs))


def test_shadow_policies_share_action_contract(shadow_env):
    ref_action = shadow_env.sample_action()
    obs = TensorDict({"render": torch.rand(3, 64, 64)}, batch_size=[])

    policies = [
        ShadowHandRandomPolicy(),
        ShadowHandHeuristicPolicy(),
        ShadowHandNeuralPolicy(image_size=(3, 64, 64)),
    ]

    for policy in policies:
        policy.reset()
        for step in range(3):
            action = policy(obs, step=step)
            name = type(policy).__name__
            assert isinstance(action, torch.Tensor), name
            assert action.shape == ref_action.shape, f"{name}: {action.shape} != {ref_action.shape}"
            assert action.dtype == ref_action.dtype, f"{name}: {action.dtype} != {ref_action.dtype}"
            assert shadow_env.action_spec.contains(action), name


def test_shadow_hand_show_target_option():
    env_hidden = ShadowHandEnv(task="block", image_size=(64, 64), show_target=False)
    env_shown = ShadowHandEnv(task="block", image_size=(64, 64), show_target=True)
    try:
        m_hidden = env_hidden._gym_env.unwrapped.model
        m_shown = env_shown._gym_env.unwrapped.model

        target_geom_id = m_hidden.geom("target").id
        target_mat_id = m_hidden.mat("material:target").id
        assert m_hidden.geom_rgba[target_geom_id, 3] == 0.0
        assert m_hidden.mat_rgba[target_mat_id, 3] == 0.0
        assert m_shown.mat_rgba[target_mat_id, 3] > 0.0
    finally:
        env_hidden.close()
        env_shown.close()


def test_shadow_hand_dataset_config():
    cfg = load_config("dataset/shadow_hand")
    assert cfg.data_module.collector.env._target_ == "wiskers.envs.shadow_hand.env.ShadowHandEnv"
    assert cfg.data_module.collector.env.show_target is False
    assert (
        cfg.data_module.collector.controller._target_
        == "wiskers.envs.shadow_hand.policies.ShadowHandHeuristicPolicy"
    )

