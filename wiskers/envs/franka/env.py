"""
FrankaKitchen-v1 environment with visual camera observations.
"""

from typing import Any, Sequence

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
from tensordict import TensorDict

from wiskers.envs.torch_env import TorchGymEnv


mujoco.mjtJoint.__eq__ = lambda self, other: (
    isinstance(other, (int, np.integer, mujoco.mjtJoint)) and int(self) == int(other)
)

AVAILABLE_TASKS: tuple[str, ...] = (
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
)


class FrankaKitchenEnv(TorchGymEnv):
    """
    FrankaKitchen-v1 environment emitting PyTorch uint8 RGB camera observations.

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
        tasks_to_complete: Subtasks for the robot to complete. Defaults to ``('microwave',)``.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        tasks_to_complete: Sequence[str] | None = ("microwave",),
    ) -> None:
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
        super().__init__(env, image_size=image_size)
        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        h, w = self.image_size or (
            getattr(renderer, "height", 480),
            getattr(renderer, "width", 480),
        )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def _process_obs(self, raw_obs: Any) -> TensorDict:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("FrankaKitchen render() returned None. Ensure render_mode='rgb_array'.")
        return super()._process_obs(np.asarray(frame, dtype=np.uint8))
