"""
Shadow Dexterous Hand environments with visual camera observations.
"""

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.torch_env import TorchGymEnv
from wiskers.envs.utils import render_to_tensor


mujoco.mjtJoint.__eq__ = lambda self, other: (
    isinstance(other, (int, np.integer, mujoco.mjtJoint)) and int(self) == int(other)
)

AVAILABLE_TASKS: dict[str, str] = {
    "reach": "HandReach-v3",
    "block": "HandManipulateBlock-v1",
    "egg": "HandManipulateEgg-v1",
    "pen": "HandManipulatePen-v1",
}


class ShadowHandEnv(TorchGymEnv):
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
    """

    def __init__(
        self,
        task: str = "block",
        image_size: tuple[int, int] | None = None,
        show_target: bool = False,
    ) -> None:
        gym.register_envs(gymnasium_robotics)

        task_key = task.lower().strip()
        if task_key not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS.keys())}")

        self.image_size = tuple(image_size) if image_size is not None else None
        self.task = task_key
        self.show_target = show_target

        env = gym.make(
            AVAILABLE_TASKS[task_key],
            render_mode="rgb_array",
        )
        super().__init__(env)

        if not show_target:
            self._hide_target_visuals()

    def _hide_target_visuals(self) -> None:
        """Set alpha=0 on target geoms, materials, and sites so goal markers are hidden in renders."""
        model: Any = getattr(self._gym_env.unwrapped, "model", None)
        if model is None:
            return
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
        frame = self._gym_env.render()
        if frame is None:
            raise RuntimeError("ShadowHand render() returned None. Ensure render_mode='rgb_array'.")
        obs_vec = torch.from_numpy(np.ascontiguousarray(raw_obs["observation"], dtype=np.float32))
        achieved = torch.from_numpy(np.ascontiguousarray(raw_obs["achieved_goal"], dtype=np.float32))
        desired = torch.from_numpy(np.ascontiguousarray(raw_obs["desired_goal"], dtype=np.float32))

        data: dict[str, torch.Tensor] = {
            "render": render_to_tensor(np.asarray(frame, dtype=np.uint8), self.image_size),
            "state:hand_qpos": obs_vec[0:24],
            "state:hand_qvel": obs_vec[24:48],
            "state:achieved_goal": achieved,
            "state:desired_goal": desired,
        }
        if self.task != "reach":
            data["state:obj_qvel"] = obs_vec[48:54]

        return TensorDict(data, batch_size=[])
