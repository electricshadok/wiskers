import pytest
import torch

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
    diffuser = VAEModule(**config_module)

    in_channels = config_module["in_channels"]
    image_size = config_module["image_size"]
    batch_size = 16
    img_data = torch.randn(batch_size, in_channels, image_size, image_size)
    labels = torch.zeros(batch_size, dtype=torch.long)
    batch = (img_data, labels)
    batch_idx = 0
    diffuser.training_step(batch, batch_idx)
