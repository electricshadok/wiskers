"""
Policies for AdroitHand dexterous manipulation environments.
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from wiskers.common.policy import ContinuousPolicy


class AdroitHandPolicy(ContinuousPolicy):
    """Base policy for AdroitHand. Continuous action space (24..30 DoF depending on task)."""

    action_dim: int = 28

    def validate_env(self, env: gym.Env) -> None:
        if not isinstance(env.action_space, gym.spaces.Box) or not env.action_space.shape:
            raise ValueError(
                f"{self.__class__.__name__} requires a continuous Box action_space, "
                f"got {env.action_space}."
            )
        self.action_dim = env.action_space.shape[0]


class AdroitHandRandomPolicy(AdroitHandPolicy):
    """Uniform random exploration for AdroitHand."""

    def forward(self, obs: np.ndarray, step: int = 0) -> np.ndarray:
        return self.sample_random()


class AdroitHandHeuristicPolicy(AdroitHandPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for AdroitHand control.

    Generates coherent reaching and finger-articulation trajectories rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
        env: Optional Gymnasium environment instance.
    """

    def __init__(
        self,
        momentum: float = 0.85,
        noise_scale: float = 0.25,
        env: Optional[gym.Env] = None,
    ):
        self.current_action: np.ndarray = np.zeros(self.action_dim, dtype=np.float32)
        super().__init__(env=env)
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.current_action = np.zeros(self.action_dim, dtype=np.float32)

    def forward(self, obs: np.ndarray, step: int = 0) -> np.ndarray:
        if self.current_action.shape[0] != self.action_dim:
            self.current_action = np.zeros(self.action_dim, dtype=np.float32)
        noise = np.random.randn(self.action_dim) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return self.current_action.copy()
