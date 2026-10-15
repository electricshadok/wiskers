"""
CarRacing → ReplayBuffer collector.

Provides a single helper function that rolls out one episode of CarRacing-v3
with the heuristic steering policy and pushes the resulting Episode into a
ReplayBuffer.

Typical usage (seed episodes before training starts):

    from wiskers.datasets.gym.car_racing_collector import collect_car_racing_episodes
    from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=200, chunk_len=50)
    collect_car_racing_episodes(buffer, n_episodes=10, max_steps=500, seed=0)

    # Then hand the buffer to TrajectoryDataModule
    dm = TrajectoryDataModule(buffer, batch_size=16, steps_per_epoch=200)
"""

import numpy as np

from wiskers.datasets.gym.car_racing import NoZoomCarRacing
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


def _collect_one_episode(env: NoZoomCarRacing, max_steps: int, seed: int) -> Episode:
    """
    Run a single CarRacing episode with the Ornstein-Uhlenbeck heuristic policy.

    Args:
        env      : An already-constructed NoZoomCarRacing instance.
        max_steps: Maximum number of environment steps per episode.
        seed     : Seed passed to env.reset() for reproducibility.

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

    steering = 0.0
    done = False
    for _ in range(max_steps):
        if done:
            break
        # OU-style mean-reverting random steering keeps the car on track
        steering = steering * 0.8 + np.random.uniform(-0.2, 0.2)
        steering = float(np.clip(steering, -1.0, 1.0))
        gas = float(np.random.uniform(0.4, 0.8))
        brake = float(np.random.uniform(0.0, 0.05))
        action = np.array([steering, gas, brake], dtype=np.float32)

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_res

        assert isinstance(next_obs, np.ndarray), f"Expected np.ndarray observation from step, got {type(next_obs)}"
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
        n_episodes : Number of episodes to collect.
        max_steps  : Maximum steps per episode.
        seed       : Base RNG seed; each episode gets seed + episode_index.
        verbose    : Print progress to stdout.
    """
    env = NoZoomCarRacing(render_mode="rgb_array")
    try:
        for i in range(n_episodes):
            episode = _collect_one_episode(env, max_steps=max_steps, seed=seed + i)
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
