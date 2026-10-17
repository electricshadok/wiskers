"""
CarRacing → ReplayBuffer collector.

Provides helper functions to roll out episodes of CarRacing-v3 with
a given CarRacingPolicy, pushing resulting Episodes into a ReplayBuffer.
"""

import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

from wiskers.datasets.gym.car_racing_policies import CarRacingPolicy
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


class NoZoomCarRacing(CarRacing):
    """
    Subclass of CarRacing that locks the camera zoom immediately to gameplay scale (t=1.0),
    bypassing the initial zooming-in animation on reset.
    """

    def reset(self, **kwargs):
        obs_info = super().reset(**kwargs)
        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            obs, info = obs_info
        else:
            obs, info = obs_info, {}

        # Lock camera zoom immediately to standard gameplay scale (bypass zoom-in animation)
        self.t = 1.0
        obs = self._render("state_pixels")

        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            return obs, info
        return obs


def _collect_one_episode(
    env: NoZoomCarRacing,
    max_steps: int,
    seed: int,
    policy: CarRacingPolicy,
) -> Episode:
    """
    Run a single CarRacing episode with the specified policy.

    Args:
        env      : An already-constructed NoZoomCarRacing instance.
        max_steps: Maximum number of environment steps per episode.
        seed     : Seed passed to env.reset() for reproducibility.
        policy   : Policy used to select actions.

    Returns:
        Episode with arrays of shape (T, H, W, C), (T, 3), (T,), (T,).
    """
    reset_res = env.reset(seed=seed)
    obs, _ = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
    assert isinstance(obs, np.ndarray), f"Expected np.ndarray observation from reset, got {type(obs)}"

    observations: list[np.ndarray] = [obs]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    policy.reset()
    done = False
    for step in range(max_steps):
        if done:
            break

        action = policy(obs, env, step)

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_res

        assert isinstance(next_obs, np.ndarray), f"Expected np.ndarray observation from step, got {type(next_obs)}"
        obs = next_obs
        observations.append(next_obs)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)

    # observations has one extra frame at the front; trim to match action length
    T = len(actions)
    return Episode(
        obs=np.array(observations[:T], dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def collect_car_racing_episodes(
    buffer: ReplayBuffer,
    policy: CarRacingPolicy,
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> None:
    """
    Collect ``n_episodes`` CarRacing rollouts and push them into ``buffer``.

    Episodes shorter than ``buffer.chunk_len`` are silently skipped by the
    buffer itself.

    Args:
        buffer     : Target ReplayBuffer to populate.
        policy     : Policy instance to collect rollouts with.
        n_episodes : Number of episodes to collect.
        max_steps  : Maximum steps per episode.
        seed       : Base RNG seed; each episode gets seed + episode_index.
        verbose    : Print progress to stdout.
    """
    env = NoZoomCarRacing(render_mode="rgb_array")
    try:
        for i in range(n_episodes):
            episode = _collect_one_episode(
                env=env,
                max_steps=max_steps,
                seed=seed + i,
                policy=policy,
            )
            buffer.add(episode)
            if verbose:
                print(
                    f"[CarRacing] episode {i + 1}/{n_episodes} | "
                    f"T={len(episode)} steps | "
                    f"total_reward={episode.rewards.sum():.1f} | "
                    f"buffer={buffer}"
                )
    finally:
        env.close()
