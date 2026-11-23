"""
Single-action sweep utilities for inspecting how each action affects an environment.
"""

import torch
from tensordict import TensorDict

from wiskers.data.episode import Episode
from wiskers.envs.action_spec import BoundedActionSpec, DiscreteActionSpec
from wiskers.envs.gym_env import GymEnv


SWEEP_NUM_STEPS = 10


def _rollout_actions(
    env: GymEnv,
    actions: list[torch.Tensor],
    seed: int,
) -> Episode:
    """Execute a fixed sequence of actions from a deterministic reset state."""
    obs, _ = env.reset(seed=seed)
    observations: list[TensorDict] = [obs]
    taken_actions: list[torch.Tensor] = []
    rewards: list[float] = []
    terminated_list: list[bool] = []
    truncated_list: list[bool] = []

    for action in actions:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(next_obs)
        taken_actions.append(action)
        rewards.append(reward)
        terminated_list.append(terminated)
        truncated_list.append(truncated)
        if terminated or truncated:
            break

    return Episode(
        obs=torch.stack(observations),
        action=torch.stack(taken_actions),
        reward=torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
        terminated=torch.tensor(terminated_list, dtype=torch.bool).unsqueeze(-1),
        truncated=torch.tensor(truncated_list, dtype=torch.bool).unsqueeze(-1),
    )


def collect_action_sweep(
    env: GymEnv,
    num_steps: int = SWEEP_NUM_STEPS,
    seed: int = 0,
) -> list[Episode]:
    """
    Collect one rollout episode per action dimension (continuous) or discrete action.

    - ``BoundedActionSpec``: For each dimension ``i``, linearly sweeps ``num_steps`` values
      from ``low[i]`` to ``high[i]`` while holding all other dimensions at a neutral
      baseline ``zeros.clamp(low, high)``.
    - ``DiscreteActionSpec``: For each action index ``i in 0..n-1``, repeats action ``i``
      for ``num_steps`` steps.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    spec = env.action_spec
    if isinstance(spec, BoundedActionSpec):
        baseline = torch.zeros_like(spec.low).clamp(spec.low, spec.high)
        episodes: list[Episode] = []
        for dim_idx in range(spec.action_dim):
            sweep_values = torch.linspace(
                float(spec.low[dim_idx]),
                float(spec.high[dim_idx]),
                num_steps,
                dtype=spec.dtype,
                device=spec.low.device,
            )
            actions = []
            for val in sweep_values:
                action = baseline.clone()
                action[dim_idx] = val
                actions.append(action)
            episodes.append(_rollout_actions(env, actions, seed=seed))
        return episodes

    if isinstance(spec, DiscreteActionSpec):
        episodes = []
        for action_idx in range(spec.n):
            action = torch.tensor([action_idx], dtype=spec.dtype)
            episodes.append(_rollout_actions(env, [action] * num_steps, seed=seed))
        return episodes

    raise ValueError(f"Unsupported action spec: {type(spec).__name__}")
