"""
Shadow Dexterous Hand environments with visual camera observations.
"""

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.mujoco_env import MujocoEnv, MujocoPhysics
from wiskers.envs.utils import patch_mujoco_joint_eq


AVAILABLE_TASKS: dict[str, str] = {
    "reach": "HandReach-v3",
    "block": "HandManipulateBlock-v1",
    "egg": "HandManipulateEgg-v1",
    "pen": "HandManipulatePen-v1",
}


class ShadowHandEnv(MujocoEnv):
    """
    Shadow Dexterous Hand environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:hand_qpos"`` (float32[24]): Hand and wrist joint angles.
        - ``"state:hand_qvel"`` (float32[24]): Hand and wrist joint velocities.
        - ``"state:obj_qvel"`` (float32[6]): Object linear/angular velocity (manipulation tasks only).
        - ``"state:achieved_goal"`` (float32[G]): Current object pose (``G=7``) or fingertip positions (``G=15``).
        - ``"state:desired_goal"`` (float32[G]): Target object pose (``G=7``) or fingertip positions (``G=15``).

    Reference:
        https://robotics.farama.org/envs/shadow_dexterous_hand/

    Args:
        task: Shadow Hand task ('reach', 'block', 'egg', 'pen'). Defaults to 'block'.
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
        show_target: Whether to render the translucent goal/target overlay in camera frames.
            Defaults to ``False``.
        physics: Optional :class:`MujocoPhysics` configuration override. Defaults to ``None``.
    """

    def __init__(
        self,
        task: str = "block",
        image_size: tuple[int, int] | None = None,
        show_target: bool = False,
        physics: MujocoPhysics | None = None,
    ) -> None:
        patch_mujoco_joint_eq()
        gym.register_envs(gymnasium_robotics)

        task_key = task.lower().strip()
        if task_key not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS.keys())}")

        self.task = task_key
        self.show_target = show_target

        env = gym.make(
            AVAILABLE_TASKS[task_key],
            render_mode="rgb_array",
        )
        super().__init__(env, image_size=image_size, physics=physics)

        if not show_target:
            self._hide_target_visuals()

    def _hide_target_visuals(self) -> None:
        """Set alpha=0 on target geoms, materials, and sites so goal markers are hidden in renders."""
        model = self.model
        for i in range(model.ngeom):
            if "target" in model.geom(i).name:
                model.geom_rgba[i, 3] = 0.0
        for i in range(model.nmat):
            if "target" in model.mat(i).name:
                model.mat_rgba[i, 3] = 0.0
        for i in range(model.nsite):
            if "target" in model.site(i).name:
                model.site_rgba[i, 3] = 0.0

    def _process_obs(self, raw_obs: dict[str, Any]) -> TensorDict:
        obs_vec = torch.from_numpy(np.ascontiguousarray(raw_obs["observation"], dtype=np.float32))
        achieved = torch.from_numpy(np.ascontiguousarray(raw_obs["achieved_goal"], dtype=np.float32))
        desired = torch.from_numpy(np.ascontiguousarray(raw_obs["desired_goal"], dtype=np.float32))

        data: dict[str, torch.Tensor] = {
            "render": self._render_frame(),
            "state:hand_qpos": obs_vec[0:24],
            "state:hand_qvel": obs_vec[24:48],
            "state:achieved_goal": achieved,
            "state:desired_goal": desired,
        }
        if self.task != "reach":
            data["state:obj_qvel"] = obs_vec[48:54]

        return TensorDict(data, batch_size=[])
