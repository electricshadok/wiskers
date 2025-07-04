import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset

from wiskers.vae.vae_module import VAEModule


@pytest.fixture
def config_module():
    config = {
        # Model Configuration
        "in_channels": 3,
        "out_channels": 3,
        "num_heads": 8,
        "image_size": 32,
        # Optimizer Configuration
        "learning_rate": 1e-4,
    }
    return config


def test_training_vae_module(config_module):
    vae = VAEModule(**config_module)

    in_channels = config_module["in_channels"]
    image_size = config_module["image_size"]
    batch_size = 8

    images = torch.randn(batch_size, in_channels, image_size, image_size)
    labels = torch.zeros(batch_size, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(vae, loader)
