"""
ALE/Breakout environment factory.
"""

import ale_py
import gymnasium as gym


# Register all ALE (Atari) environments
gym.register_envs(ale_py)


def make_breakout_env() -> gym.Env:
    """
    Construct an ALE/Breakout-v5 environment emitting raw RGB observations.

    Returns:
        Gymnasium environment instance.
    """
    return gym.make("ALE/Breakout-v5", render_mode="rgb_array")
