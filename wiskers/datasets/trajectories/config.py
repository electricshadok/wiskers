from dataclasses import dataclass


@dataclass
class CollectionConfig:
    """
    Configuration for environment rollout collection.

    Args:
        seed_episodes (int): Initial rollouts collected before training starts.
        max_steps_per_episode (int): Step cutoff per environment episode.
        steps_per_iteration (int): Number of steps to collect per online training loop.
    """

    seed_episodes: int = 20
    max_steps_per_episode: int = 500
    steps_per_iteration: int = 2000
