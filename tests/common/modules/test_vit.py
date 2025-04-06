import pytest
import torch

from wiskers.common.modules.vit import PatchEmbedding


@pytest.mark.parametrize(
    "batch_size, in_channels, height, width",
    [
        (4, 8, 32, 32),
        (8, 8, 16, 16),
    ],
)
def test_transformer_attention_2d(batch_size, in_channels, height, width):
    patch_size = 8
    embed_dim = 16
    patcher = PatchEmbedding(patch_size=8, in_channels=in_channels, embed_dim=embed_dim)
    x = torch.randn(batch_size, in_channels, height, width)

    out_x = patcher(x)

    num_patches = (height / patch_size) * (width / patch_size)
    assert out_x.shape == (batch_size, num_patches, embed_dim)
