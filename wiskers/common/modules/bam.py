import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Implementation of channel attention mechanism from BAM
    BAM: Bottleneck Attention Module: https://arxiv.org/abs/1807.06514

    Shapes:
        Input: (N, C, H, W)
        Output; (N, C, 1, 1)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        hidden_channels = in_channels // reduction_ratio

        if hidden_channels < 1:
            raise ValueError(f"Reduction ratio {reduction_ratio} is too high for {in_channels} input channels.")

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(fc1, fc2)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.global_avg_pool(x)
        out = self.mlp(out)
        out = self.bn(out)
        return out


class SpatialAttention(nn.Module):
    """
    Implementation of spatial attention mechanism from BAM
    BAM: Bottleneck Attention Module: https://arxiv.org/abs/1807.06514

    Shapes:
        Input: (N, C, H, W)
        Output; (N, 1, H, W)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4, dilation: int = 4):
        super().__init__()
        hidden_channels = in_channels // reduction_ratio

        if hidden_channels < 1:
            raise ValueError(f"Reduction ratio {reduction_ratio} is too high for {in_channels} input channels.")

        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.spatial_atts(x)
        out = self.bn(out)
        return out


class BAM(nn.Module):
    """
    Bottleneck Attention Module (BAM)
    Paper: https://arxiv.org/abs/1807.06514
    F' = F + F x sigma(M_c(F) + M_s(F)) where
        - M_c is the channel attention
        - M_s is the spatial attention
    ...

    Input:  (N, C, H, W)
    Output: (N, C, H, W)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4, dilation: int = 4):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(in_channels, reduction_ratio, dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mc = self.channel_att(x)  # (N, C, 1, 1)
        ms = self.spatial_att(x)  # (N, 1, H, W)

        mc = mc.expand_as(x)  # (N, C, H, W)
        ms = ms.expand_as(x)  # (N, C, H, W)

        att = self.sigmoid(mc + ms)  # Combine + sigmoid
        return x + x * att  # Residual connection
