"""
InvertedPendulum-v5 MuJoCo environment with visual camera observations.
"""

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.gym_env import ObsFieldTransform
from wiskers.envs.mujoco_env import MujocoEnv, MujocoPhysics


class InvertedPendulumEnv(MujocoEnv):
    """
    InvertedPendulum-v5 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:qpos"`` (float32[2]): ``[cart_pos (m), pole_angle (rad)]``.
        - ``"state:qvel"`` (float32[2]): ``[cart_vel (m/s), pole_angular_vel (rad/s)]``.

    Reference:
        https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

    Args:
        reset_noise_scale: Scale of uniform random perturbations added to initial
            positions and velocities on reset. Must be non-negative. Defaults to ``0.01``.
        physics: Optional :class:`MujocoPhysics` configuration override. Defaults to ``None``.
        obs_fields: Optional mapping from observation field name to per-field transform
            callable (or ``None`` to keep unchanged). Defaults to ``None``.
    """

    obs_keys: tuple[str, ...] = ("render", "state:qpos", "state:qvel")

    def __init__(
        self,
        reset_noise_scale: float = 0.01,
        physics: MujocoPhysics | None = None,
        obs_fields: Mapping[str, ObsFieldTransform | None] | None = None,
    ) -> None:
        if reset_noise_scale < 0:
            raise ValueError(f"reset_noise_scale must be non-negative, got {reset_noise_scale}.")

        self.reset_noise_scale = reset_noise_scale
        env = gym.make(
            "InvertedPendulum-v5",
            render_mode="rgb_array",
            reset_noise_scale=reset_noise_scale,
        )
        super().__init__(env, physics=physics, obs_fields=obs_fields)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        state = torch.from_numpy(np.ascontiguousarray(raw_obs, dtype=np.float32))
        return TensorDict(
            {
                "render": self._render_frame(),
                "state:qpos": state[0:2],
                "state:qvel": state[2:4],
            },
            batch_size=[],
        )
