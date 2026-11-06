"""
AdroitHand-v1 environment factory with visual camera observations.
"""

from typing import Any, Optional

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np


mujoco.mjtJoint.__eq__ = lambda self, other: isinstance(
    other, (int, np.integer, mujoco.mjtJoint)
) and int(self) == int(other)

AVAILABLE_TASKS: dict[str, str] = {
    "door": "AdroitHandDoor-v1",
    "hammer": "AdroitHandHammer-v1",
    "pen": "AdroitHandPen-v1",
    "relocate": "AdroitHandRelocate-v1",
}


class AdroitHandVisualObservation(gym.ObservationWrapper):
    """
    Observation wrapper for AdroitHand that extracts RGB camera renders as observations.

    Guarantees that observations returned by reset() and step() are (H, W, 3) uint8 arrays.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w = 480, 480
        if hasattr(env.unwrapped, "mujoco_renderer"):
            h = getattr(env.unwrapped.mujoco_renderer, "height", 480)
            w = getattr(env.unwrapped.mujoco_renderer, "width", 480)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

    def observation(self, observation: Any) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("AdroitHand render() returned None. Ensure render_mode='rgb_array'.")
        return np.asarray(frame, dtype=np.uint8)


def make_adroit_hand_env(
    task: str = "door",
    image_size: Optional[tuple[int, int]] = (64, 64),
) -> gym.Env:
    """
    Construct an AdroitHand-v1 environment with camera visual observations.

    Args:
        task: Manipulation task for the Adroit dexterous hand. Valid tasks:
              'door', 'hammer', 'pen', 'relocate'. Defaults to 'door'.
        image_size: Target (height, width) to resize frames to. If None, native
                    resolution (480, 480) is preserved.

    Returns:
        Gymnasium environment instance emitting visual RGB frames.
    """
    gym.register_envs(gymnasium_robotics)

    task_key = task.lower().strip()
    if task_key not in AVAILABLE_TASKS:
        raise ValueError(f"Unknown task '{task}'. Available tasks: {list(AVAILABLE_TASKS.keys())}")

    env = gym.make(
        AVAILABLE_TASKS[task_key],
        render_mode="rgb_array",
    )
    env = AdroitHandVisualObservation(env)
    if image_size is not None:
        env = gym.wrappers.ResizeObservation(env, tuple(image_size))
    return env

