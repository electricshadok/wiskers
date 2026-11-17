"""
AdroitHand-v1 environment with visual camera observations.
"""

from typing import Any

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import torch

from wiskers.envs.torch_env import TorchGymEnv


mujoco.mjtJoint.__eq__ = lambda self, other: (
    isinstance(other, (int, np.integer, mujoco.mjtJoint)) and int(self) == int(other)
)

AVAILABLE_TASKS: dict[str, str] = {
    "door": "AdroitHandDoor-v1",
    "hammer": "AdroitHandHammer-v1",
    "pen": "AdroitHandPen-v1",
    "relocate": "AdroitHandRelocate-v1",
}


class AdroitHandEnv(TorchGymEnv):
    """
    AdroitHand-v1 environment emitting PyTorch uint8 RGB camera observations.

    Args:
        task: Manipulation task ('door', 'hammer', 'pen', 'relocate'). Defaults to 'door'.
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(480, 480)`` resolution.
    """

    def __init__(
        self,
        task: str = "door",
        image_size: tuple[int, int] | None = None,
    ) -> None:
        gym.register_envs(gymnasium_robotics)

        task_key = task.lower().strip()
        if task_key not in AVAILABLE_TASKS:
            raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS.keys())}")

        env = gym.make(
            AVAILABLE_TASKS[task_key],
            render_mode="rgb_array",
        )
        super().__init__(env, image_size=image_size)
        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        h, w = self.image_size or (
            getattr(renderer, "height", 480),
            getattr(renderer, "width", 480),
        )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

    def _process_obs(self, raw_obs: Any) -> torch.Tensor:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("AdroitHand render() returned None. Ensure render_mode='rgb_array'.")
        return super()._process_obs(np.asarray(frame, dtype=np.uint8))
