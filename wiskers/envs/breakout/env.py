"""
ALE/Breakout environment.
"""

import ale_py
import gymnasium as gym

from wiskers.envs.torch_env import TorchGymEnv


# Register all ALE (Atari) environments
gym.register_envs(ale_py)


class BreakoutEnv(TorchGymEnv):
    """
    ALE/Breakout-v5 environment emitting PyTorch uint8 RGB observations.

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(210, 160)`` resolution.
    """

    def __init__(self, image_size: tuple[int, int] | None = None) -> None:
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        super().__init__(env, image_size=image_size)
