import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=1,
        dilation=1,
        activation=F.relu,
    ):
        """
        Implementation of a Gated Convolution in 2D.
        Uses a gating mechanism to dynamically control information flow, improving feature selection.
        Paper: "Free-Form Image Inpainting with Gated Convolution", 2019

        Shape:
            Input: (N, in_channels, H, W)
            Output: (N, out_channels, H, W)
        """
        super().__init__()

        self.activation = activation

        # Standard convolution to extract features
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

        # Gating convolution to control information flow
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        # Compute feature map
        feature_map = self.activation(self.feature_conv(x))

        # Compute gate (sigmoid to ensure values between 0 and 1)
        gate = torch.sigmoid(self.gate_conv(x))

        # Element-wise multiplication (gating mechanism)
        return feature_map * gate
