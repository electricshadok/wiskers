import torch.nn as nn


class CausalConv3d(nn.Module):
    """
    Implementation of a causal convolution in 3D.
    Ensures that the output at (t, h, w) depends only on input elements from (t, h, w) and earlier.

    Shape:
        Input: (N, in_channels, D, H, W)
        Output: (N, out_channels, D, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        dilation: tuple = (1, 1, 1),
        bias: bool = False,
    ):
        super().__init__()

        # Compute padding only for front (depth), top (height), and left (width)
        self.pad_d = (kernel_size[0] - 1) * dilation[0]  # Temporal padding
        self.pad_h = (kernel_size[1] - 1) * dilation[1]  # Height padding
        self.pad_w = (kernel_size[2] - 1) * dilation[2]  # Width padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.pad_d, self.pad_h, self.pad_w),  # Applies padding on all sides
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x[:, :, : -self.pad_d, : -self.pad_h, : -self.pad_w]  # Remove future values
