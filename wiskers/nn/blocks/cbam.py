import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Implementation of channel attention mechanism from CBAM
    Paper: https://arxiv.org/abs/1807.06521v2

    Shapes:
        Input: (N, in_C, H, W)
        Output; (N, in_C, 1, 1)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        hidden_channels = in_channels // reduction_ratio
        if hidden_channels < 1:
            raise ValueError(f"Reduction ratio {reduction_ratio} is too high for {in_channels} input channels.")

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        relu = nn.ReLU()
        fc2 = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.shared_mlp = nn.Sequential(fc1, relu, fc2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Implementation of spatial attention from CBAM
    Paper: https://arxiv.org/abs/1807.06521v2
    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, 1, H, W)
    """

    def __init__(self):
        super().__init__()
        kernel_size, padding = 7, 3  # default values from the papaer
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply avg pooling and max pooling on the input feature map across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


class CBAM(nn.Module):
    """
    Implementation of Convolutional Block Attention Module (CBAM)
    Paper: https://arxiv.org/abs/1807.06521v2

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, in_C, H, W)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x
