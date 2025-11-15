import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset

from wiskers.modules.gan_module import GANModule


@pytest.fixture
def config_module():
    config = {
        # Model Configuration
        "in_channels": 3,
        "image_size": 16,
        "num_classes": 10,
        # Optimizer Configuration
        "learning_rate": 1e-4,
    }
    return config


def test_training_gan_module(config_module):
    gan = GANModule(**config_module)

    in_channels = config_module["in_channels"]
    image_size = config_module["image_size"]
    num_classes = config_module["num_classes"]
    batch_size = 8

    images = torch.randn(batch_size, in_channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(gan, loader)
