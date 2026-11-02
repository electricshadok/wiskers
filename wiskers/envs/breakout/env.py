"""
ALE/Breakout environment factory.
"""

from typing import Optional

import ale_py
import gymnasium as gym


# Register all ALE (Atari) environments
gym.register_envs(ale_py)


def make_breakout_env(image_size: Optional[tuple[int, int]] = (64, 64)) -> gym.Env:
    """
    Construct an ALE/Breakout-v5 environment with optional observation resizing.

    Args:
        image_size: Target (height, width) to resize frames to. If None, native
                    resolution (210, 160) is preserved.

    Returns:
        Gymnasium environment instance.
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    if image_size is not None:
        env = gym.wrappers.ResizeObservation(env, tuple(image_size))
    return env

