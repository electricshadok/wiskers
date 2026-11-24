"""
FrankaKitchen-v1 environment with visual camera observations.
"""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.mujoco_env import MujocoEnv
from wiskers.envs.utils import patch_mujoco_joint_eq


AVAILABLE_TASKS: tuple[str, ...] = (
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
)


class FrankaKitchenEnv(MujocoEnv):
    """
    FrankaKitchen-v1 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:robot_qpos"`` (float32[9]): ``observation[0:9]`` — 7 Panda arm joint
          angles (rad) + 2 gripper finger translations (m).
        - ``"state:robot_qvel"`` (float32[9]): ``observation[9:18]`` — 7 Panda arm joint
          angular velocities (rad/s) + 2 gripper finger linear velocities (m/s).
        - ``"state:obj_qpos"`` (float32[21]): ``observation[18:39]`` — kitchen object
          positions/orientations (4 burners [8], light switch [2], slide cabinet [1],
          2 hinge cabinets [2], microwave door [1], kettle (x, y, z) [3] +
          quaternion (qx, qy, qz, qw) [4]).
        - ``"state:obj_qvel"`` (float32[20]): ``observation[39:59]`` — kitchen object
          velocities (4 burners [8], light switch [2], slide cabinet [1],
          2 hinge cabinets [2], microwave door [1], kettle linear [3] + angular [3]
          velocity).

    Reference:
        https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
        tasks_to_complete: Subtasks for the robot to complete. Defaults to ``('microwave',)``.
        gravity: Optional ``(gx, gy, gz)`` gravity vector override in m/s². Defaults to ``None``.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        tasks_to_complete: Sequence[str] | None = ("microwave",),
        gravity: Sequence[float] | None = None,
    ) -> None:
        patch_mujoco_joint_eq()
        gym.register_envs(gymnasium_robotics)

        tasks = list(tasks_to_complete) if tasks_to_complete is not None else ["microwave"]
        for task in tasks:
            if task not in AVAILABLE_TASKS:
                raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS)}")

        env = gym.make(
            "FrankaKitchen-v1",
            tasks_to_complete=tasks,
            render_mode="rgb_array",
        )
        super().__init__(env, image_size=image_size, gravity=gravity)

    def _process_obs(self, raw_obs: dict[str, Any]) -> TensorDict:
        state = torch.from_numpy(np.ascontiguousarray(raw_obs["observation"], dtype=np.float32))
        return TensorDict(
            {
                "render": self._render_frame(),
                "state:robot_qpos": state[0:9],
                "state:robot_qvel": state[9:18],
                "state:obj_qpos": state[18:39],
                "state:obj_qvel": state[39:59],
            },
            batch_size=[],
        )
