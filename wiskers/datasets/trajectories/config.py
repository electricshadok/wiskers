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


@dataclass
class TrainingConfig:
    """
    Configuration for batch training schedule and DataLoader.

    Args:
        batch_size (int): Number of sequence chunks per training batch.
        steps_per_epoch (int): Number of batches to yield per epoch.
        num_workers (int): DataLoader worker processes (default: 0).
    """

    batch_size: int = 32
    steps_per_epoch: int = 1000
    num_workers: int = 0
