"""
Common base policy interface for environment rollouts and learned agents.
"""

from abc import ABC, abstractmethod
from typing import Union

import gymnasium as gym
import numpy as np
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for all rollout policies (random, heuristic, or learned).

    Attributes:
        action_dim (int): Dimensionality of the stored action vector in Episode.actions
            (1 for discrete scalar actions, or D for D-dimensional continuous actions).
        action_type (str): Either ``"discrete"`` or ``"continuous"``.
    """

    action_dim: int
    action_type: str

    def reset(self) -> None:
        """Reset any internal recurrent or episodic state at the start of an episode."""
        pass

    @abstractmethod
    def forward(
        self,
        obs: np.ndarray,
        env: gym.Env,
        step: int = 0,
    ) -> Union[int, np.ndarray]:
        """
        Compute the environment action for the current step.

        Args:
            obs (np.ndarray): Current raw observation from the environment.
            env (gym.Env): Environment instance.
            step (int): Current step index within the episode.

        Returns:
            Union[int, np.ndarray]: Discrete action index (``int``) or continuous action array (``np.ndarray``).
        """
        raise NotImplementedError

