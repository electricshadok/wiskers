"""
ALE/Breakout environment.
"""

import ale_py
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.torch_env import TorchGymEnv


# Register all ALE (Atari) environments
gym.register_envs(ale_py)


class BreakoutEnv(TorchGymEnv):
    """
    ALE/Breakout-v5 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"ram"`` (uint8[128]): Atari 2600 console RAM state.

    Reference:
        https://ale.farama.org/environments/breakout/

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(210, 160)`` resolution.
    """

    def __init__(self, image_size: tuple[int, int] | None = None) -> None:
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        super().__init__(env, image_size=image_size)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        obs_td = super()._process_obs(raw_obs)
        ram = self.env.unwrapped.ale.getRAM()
        obs_td["ram"] = torch.from_numpy(np.ascontiguousarray(ram, dtype=np.uint8))
        return obs_td
