"""
InvertedPendulum-v5 MuJoCo environment with visual camera observations.
"""

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.gym_env import GymEnv
from wiskers.envs.utils import render_to_tensor


class InvertedPendulumEnv(GymEnv):
    """
    InvertedPendulum-v5 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:qpos"`` (float32[2]): ``[cart_pos (m), pole_angle (rad)]``.
        - ``"state:qvel"`` (float32[2]): ``[cart_vel (m/s), pole_angular_vel (rad/s)]``.

    Reference:
        https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
        reset_noise_scale: Scale of uniform random perturbations added to initial
            positions and velocities on reset. Must be non-negative. Defaults to ``0.01``.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        reset_noise_scale: float = 0.01,
    ) -> None:
        if reset_noise_scale < 0:
            raise ValueError(f"reset_noise_scale must be non-negative, got {reset_noise_scale}.")

        self.image_size = tuple(image_size) if image_size is not None else None
        self.reset_noise_scale = reset_noise_scale
        env = gym.make(
            "InvertedPendulum-v5",
            render_mode="rgb_array",
            reset_noise_scale=reset_noise_scale,
        )
        super().__init__(env)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        frame = self._gym_env.render()
        if frame is None:
            raise RuntimeError("InvertedPendulum render() returned None. Ensure render_mode='rgb_array'.")
        state = torch.from_numpy(np.ascontiguousarray(raw_obs, dtype=np.float32))
        return TensorDict(
            {
                "render": render_to_tensor(np.asarray(frame, dtype=np.uint8), self.image_size),
                "state:qpos": state[0:2],
                "state:qvel": state[2:4],
            },
            batch_size=[],
        )
