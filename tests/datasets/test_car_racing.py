import torch

from wiskers.datasets.gym.car_racing import CarRacingDataModule


def test_car_racing_datamodule_init():
    dm = CarRacingDataModule(
        capacity=100,
        chunk_len=20,
        seed_episodes=0,
        batch_size=4,
        steps_per_epoch=4,
    )
    assert dm.buffer.capacity == 100
    assert dm.buffer.chunk_len == 20
    assert dm.buffer.is_empty


def test_car_racing_datamodule_setup_and_dataloader():
    dm = CarRacingDataModule(
        capacity=10,
        chunk_len=5,
        seed_episodes=1,
        max_steps_per_episode=8,
        batch_size=2,
        steps_per_epoch=2,
        verbose=False,
    )
    dm.setup()
    assert not dm.buffer.is_empty
    assert dm.buffer.num_episodes == 1

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert set(batch.keys()) == {"obs", "actions", "rewards", "dones"}
    assert batch["obs"].shape == (2, 5, 96, 96, 3)
    assert batch["actions"].shape == (2, 5, 3)
    assert batch["rewards"].shape == (2, 5)
    assert batch["dones"].shape == (2, 5)
    assert batch["obs"].dtype == torch.float32
    assert batch["obs"].min() >= 0.0
    assert batch["obs"].max() <= 1.0
