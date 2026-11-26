"""
AdroitHand-v1 environment with visual camera observations.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.mujoco_env import MujocoEnv, MujocoPhysics


AVAILABLE_TASKS: dict[str, str] = {
    "door": "AdroitHandDoor-v1",
    "hammer": "AdroitHandHammer-v1",
    "pen": "AdroitHandPen-v1",
    "relocate": "AdroitHandRelocate-v1",
}

HAND_DIM: dict[str, int] = {
    "door": 27,
    "hammer": 26,
    "pen": 24,
    "relocate": 30,
}


class AdroitHandEnv(MujocoEnv):
    """
    AdroitHand-v1 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:hand"`` (float32[N]): Arm, wrist, and finger joint angles
          (``N=27`` door, ``26`` hammer, ``24`` pen, ``30`` relocate).
        - ``"state:obj"`` (float32[M]): Object/target state and velocities
          (``M=12`` door, ``20`` hammer, ``21`` pen, ``9`` relocate).

    Reference:
        https://robotics.farama.org/envs/adroit_hand/

    Args:
        task: Manipulation task ('door', 'hammer', 'pen', 'relocate'). Defaults to 'door'.
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
        physics: Optional :class:`MujocoPhysics` configuration override. Defaults to ``None``.
    """

    def __init__(
        self,
        task: str = "door",
        image_size: tuple[int, int] | None = None,
        physics: MujocoPhysics | None = None,
    ) -> None:
        gym.register_envs(gymnasium_robotics)

        task_key = task.lower().strip()
        if task_key not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS.keys())}")

        self.task = task_key
        self._hand_dim = HAND_DIM[task_key]

        env = gym.make(
            AVAILABLE_TASKS[task_key],
            render_mode="rgb_array",
        )
        super().__init__(env, image_size=image_size, physics=physics)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        state = torch.from_numpy(np.ascontiguousarray(raw_obs, dtype=np.float32))
        n = self._hand_dim
        return TensorDict(
            {
                "render": self._render_frame(),
                "state:hand": state[:n],
                "state:obj": state[n:],
            },
            batch_size=[],
        )
