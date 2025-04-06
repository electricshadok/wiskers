import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings using a convolutional layer.
    Each patch is flattened and projected to an embedding dimension.

    Args:
        patch_size (int): Size of each patch (assumed square).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dim (int): Dimension of the output embedding for each patch.

    Shape:
        Input: (B, in_channels, H, W)
        Output: (B, N, embed_dim), where
            N is the number of patches
            N = (H / patch_size) * (W / patch_size)
    """

    def __init__(self, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # TODO
        # Output shape: (B, P, no. of channels)
        return self.conv(x).flatten(start_dim=2).transpose(1, 2)
