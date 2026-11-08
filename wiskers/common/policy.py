"""
Common base policy interfaces for discrete and continuous environment rollouts.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for all rollout policies (random, heuristic, or learned).

    Attributes:
        action_dim (int): Dimensionality of the stored action vector in Episode.actions
            (1 for discrete scalar actions, or D for D-dimensional continuous actions).
    """

    action_dim: int

    def __init__(self, env: Optional[gym.Env] = None) -> None:
        super().__init__()
        self.env: Optional[gym.Env] = None
        if env is not None:
            self.set_env(env)

    def set_env(self, env: gym.Env) -> None:
        """Validate and attach the Gymnasium environment to this policy."""
        self._validate_env(env)
        self.env = env

    def _validate_env(self, env: gym.Env) -> None:
        """Optional hook to verify that ``env`` is compatible with this policy."""
        pass

    def reset(self) -> None:
        """Reset any internal recurrent or episodic state at the start of an episode."""
        pass

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray,
        step: int = 0,
    ) -> Union[int, np.ndarray]:
        """
        Compute the environment action for the current step.

        Args:
            obs (np.ndarray): Current raw observation from the environment.
            step (int): Current step index within the episode.

        Returns:
            Union[int, np.ndarray]: Discrete action index (``int``) or continuous action array (``np.ndarray``).
        """
        raise NotImplementedError


class DiscretePolicy(BasePolicy):
    """Base policy for discrete action spaces returning an ``int`` action index."""

    action_dim: int = 1

    def _validate_env(self, env: gym.Env) -> None:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} requires a Discrete action_space, "
                f"got {env.action_space}."
            )

    def sample_random(self) -> int:
        """Sample a uniform random discrete action index from ``self.env.action_space``."""
        if self.env is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.env is not set. Pass env to __init__ or call set_env(env)."
            )
        return int(self.env.action_space.sample())

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray,
        step: int = 0,
    ) -> int:
        raise NotImplementedError


class ContinuousPolicy(BasePolicy):
    """Base policy for continuous action spaces returning a ``float32`` ``np.ndarray``."""

    def _validate_env(self, env: gym.Env) -> None:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f"{self.__class__.__name__} requires a continuous Box action_space, "
                f"got {env.action_space}."
            )
        if env.action_space.shape != (self.action_dim,):
            raise ValueError(
                f"{self.__class__.__name__} expects action_dim={self.action_dim}, "
                f"got action_space shape {env.action_space.shape}."
            )

    def sample_random(self) -> np.ndarray:
        """Sample a uniform random continuous action vector from ``self.env.action_space``."""
        if self.env is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.env is not set. Pass env to __init__ or call set_env(env)."
            )
        return np.asarray(self.env.action_space.sample(), dtype=np.float32)

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray,
        step: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError
