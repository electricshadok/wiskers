import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset

from wiskers.diffusion.diffuser_module import DiffuserModule


@pytest.fixture
def config_module():
    config = {
        # Model Configuration
        "in_channels": 3,
        "out_channels": 3,
        "time_dim": 256,
        "num_heads": 8,
        "image_size": 32,
        # Scheduler Configuration
        "scheduler_type": "ddpm",
        "num_steps": 1000,
        "beta_start": 1e-5,
        "beta_end": 1e-2,
        "beta_schedule": "linear",
        # Optimizer Configuration
        "learning_rate": 1e-4,
    }
    return config


def test_training_diffuser_module(config_module):
    diffuser = DiffuserModule(**config_module)

    in_channels = config_module["in_channels"]
    image_size = config_module["image_size"]
    batch_size = 8

    images = torch.randn(batch_size, in_channels, image_size, image_size)
    labels = torch.zeros(batch_size, dtype=torch.long)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(diffuser, loader)


def test_sample_generation(config_module):
    diffuser = DiffuserModule(**config_module)
    num_samples = 2
    num_inference_steps = 10
    in_channels = config_module["in_channels"]
    image_size = config_module["image_size"]
    img_data = diffuser.generate_samples(num_samples, num_inference_steps)

    assert isinstance(img_data, torch.Tensor)
    assert img_data.shape == (num_samples, in_channels, image_size, image_size)
    assert img_data.dtype == torch.float32
