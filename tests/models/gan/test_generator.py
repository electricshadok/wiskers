import pytest
import torch

from wiskers.models.gan.generator import Generator


@pytest.mark.parametrize(
    "batch_size, n_channels, img_size, num_classes, image_embedding, class_embedding",
    [
        (4, 3, 32, 10, 16, 16),
        (8, 3, 16, 10, 8, 8),
    ],
)
def test_gan_generator(batch_size, n_channels, img_size, num_classes, image_embedding, class_embedding):
    filters = [32, 16, 8, n_channels]
    net = Generator(img_size, num_classes, image_embedding, class_embedding, filters)

    noise = torch.randn(batch_size, image_embedding)
    labels = torch.randint(0, num_classes, (batch_size,))

    out = net(noise, labels)

    assert out.shape == (batch_size, n_channels, img_size, img_size)
    assert out.dtype == noise.dtype
