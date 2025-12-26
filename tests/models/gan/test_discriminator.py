import pytest
import torch

from wiskers.models.gan.discriminator import Discriminator


@pytest.mark.parametrize(
    "batch_size, in_channels, img_size, num_classes, class_embedding",
    [
        (4, 3, 32, 10, 16),
        (8, 3, 16, 10, 8),
    ],
)
def test_gan_discriminator(batch_size, in_channels, img_size, num_classes, class_embedding):
    filters = [in_channels, 8, 16, 32]
    net = Discriminator(img_size, num_classes, class_embedding, filters)
    img = torch.randn(batch_size, in_channels, img_size, img_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    out = net(img, labels)

    assert out.shape == (batch_size, 1)
    assert out.dtype == img.dtype
