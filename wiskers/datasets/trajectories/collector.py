"""
Modular rollout collector for trajectory-based RL and world models.
"""

import torch

from wiskers.common.policy import BasePolicy
from wiskers.datasets.trajectories.buffer import Buffer
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.transforms import EpisodeTransform
from wiskers.envs.torch_env import TorchGymEnv


class Collector:
    """
    Rollout collector that gathers episodes from a TorchGymEnv using a policy
    and stores them in a Buffer as CPU PyTorch tensors.

    Guarantees:
        - Collects ``num_episodes`` valid episodes.
        - Enforces ``min_steps``: any episode ending earlier than ``min_steps`` is discarded.
        - Enforces ``max_steps``: clamps episodes from exceeding ``max_steps``.
        - Stores observations as compact ``uint8`` CPU tensors while passing transformed
          PyTorch tensors to the policy.

    Args:
        env (TorchGymEnv): The PyTorch-wrapped Gymnasium environment instance.
        policy (BasePolicy): Policy instance implementing ``BasePolicy``.
        transform (EpisodeTransform): Transform used to convert raw ``uint8``
            observations into normalized ``torch.Tensor`` before passing them to ``policy``.
        num_episodes (int): Target number of valid episodes to collect.
        min_steps (int): Minimum step cutoff; episodes with fewer steps are rejected.
        max_steps (int): Maximum step cutoff per episode (clamped).
        seed (int): Base RNG seed for environment resets.
        verbose (bool): Whether to log collection progress.
    """

    def __init__(
        self,
        env: TorchGymEnv,
        policy: BasePolicy,
        transform: EpisodeTransform,
        num_episodes: int = 10,
        min_steps: int = 0,
        max_steps: int = 500,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.policy.set_env(env)
        self.transform = transform
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
            Episode containing collected transitions as CPU tensors.
        """
        self.policy.reset()

        obs, _ = self.env.reset(seed=self.seed + self._episode_idx)
        self._episode_idx += 1

        observations: list[torch.Tensor] = [obs]
        actions: list[torch.Tensor] = []
        rewards: list[float] = []
        dones: list[bool] = []

        done = False
        for step in range(self.max_steps):
            if done:
                break

            obs_tensor = self.transform.transform_obs(obs)
            action = self.policy(obs_tensor, step)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            obs = next_obs
            observations.append(obs)
            actions.append(action.detach().cpu().reshape(-1).to(dtype=torch.float32))
            rewards.append(reward)
            dones.append(done)

        return Episode(
            obs=torch.stack(observations),
            actions=torch.stack(actions),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            dones=torch.tensor(dones, dtype=torch.bool),
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
                    f"reward={episode.rewards.sum().item():.1f} | "
                    f"buffer={buffer}"
                )

        return total_steps

    def close(self) -> None:
        self.env.close()
