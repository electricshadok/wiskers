import torch.nn as nn


class SEBlock(nn.Module):
    """
    Implementation of the Squeeze-and-Excitation Block (SEBlock) for channel-wise attention in CNNs.
    Paper: "Squeeze-and-excitation networks", 2017.

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for bottleneck in excitation path.

    Notes:
        Find an implementation using convolution here : https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#SqueezeExcitation
        TODO: need to compare both implementation

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze_channels),
            nn.ReLU(),
            nn.Linear(squeeze_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        # Squeeze step: Global Average Pooling (GAP)
        y = self.avg_pool(x).view(N, C)
        # Excitation step: Passing through fully connected layers
        y = self.fc(y).view(N, C, 1, 1)
        # Scale the input x with the output activations (channel-wise scaling)
        return x * y
