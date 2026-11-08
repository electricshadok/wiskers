"""
Modular rollout collector for trajectory-based RL and world models.
"""

from typing import Any

import gymnasium as gym
import numpy as np

from wiskers.common.policy import BasePolicy
from wiskers.datasets.trajectories.buffer import Buffer
from wiskers.datasets.trajectories.episode import Episode


class Collector:
    """
    Rollout collector that gathers episodes from an environment using a policy
    and stores them in a ReplayBuffer.

    Guarantees:
        - Collects ``num_episodes`` valid episodes.
        - Enforces ``min_steps``: any episode ending earlier than ``min_steps`` is discarded.
        - Enforces ``max_steps``: clamps episodes from exceeding ``max_steps``.
        - Stores observations as compact ``uint8`` numpy arrays.

    Args:
        env (gym.Env): The Gymnasium environment instance.
        policy (BasePolicy): Policy instance implementing ``BasePolicy``.
        num_episodes (int): Target number of valid episodes to collect.
        min_steps (int): Minimum step cutoff; episodes with fewer steps are rejected.
        max_steps (int): Maximum step cutoff per episode (clamped).
        seed (int): Base RNG seed for environment resets.
        verbose (bool): Whether to log collection progress.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: BasePolicy,
        num_episodes: int = 10,
        min_steps: int = 0,
        max_steps: int = 500,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.seed = seed
        self.verbose = verbose
        self._episode_idx = 0

    def collect_one_episode(self) -> Episode:
        """
        Run a single environment rollout using self.env and self.policy up to self.max_steps.

        Returns:
            Episode containing collected transitions.
        """
        self.policy.reset()

        obs, _ = self.env.reset(seed=self.seed + self._episode_idx)
        self._episode_idx += 1

        observations: list[np.ndarray] = [obs]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []

        done = False
        for step in range(self.max_steps):
            if done:
                break

            action = self.policy(obs, self.env, step)
            env_action, stored_action = self._format_action(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(env_action)
            done = bool(terminated or truncated)

            obs = next_obs
            observations.append(next_obs)
            actions.append(stored_action)
            rewards.append(float(reward))
            dones.append(done)

        return Episode(
            obs=np.array(observations, dtype=np.uint8),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            dones=np.array(dones, dtype=bool),
        )

    def collect(
        self,
        buffer: Buffer,
    ) -> int:
        """
        Collect ``self.num_episodes`` valid episodes into ``buffer``.
        Episodes shorter than ``self.min_steps`` are discarded and re-collected.

        Args:
            buffer: Target Buffer (ReplayBuffer, RolloutBuffer) to store episodes.

        Returns:
            Total environment steps collected across all accepted episodes.
        """
        collected = 0
        total_steps = 0

        while collected < self.num_episodes:
            episode = self.collect_one_episode()
            if len(episode) < self.min_steps:
                continue

            buffer.add(episode)
            collected += 1
            total_steps += len(episode)

            if self.verbose:
                print(
                    f"[Collector] episode {collected}/{self.num_episodes} | "
                    f"T={len(episode)} steps | "
                    f"reward={episode.rewards.sum():.1f} | "
                    f"buffer={buffer}"
                )

        return total_steps

    def close(self) -> None:
        self.env.close()

    @staticmethod
    def _format_action(action: Any) -> tuple[Any, np.ndarray]:
        """
        Gymnasium env.step() needs primitive types for Discrete spaces (e.g. int),
        while Episode.actions must be a float32 array of shape (action_dim,).
        """
        if isinstance(action, (int, np.integer)):
            return int(action), np.array([float(action)], dtype=np.float32)
        act_arr = np.asarray(action, dtype=np.float32)
        return act_arr, act_arr
