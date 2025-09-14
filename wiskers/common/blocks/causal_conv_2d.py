import torch.nn as nn


class CausalConv2d(nn.Module):
    """
    Implementation of a causal convolution in 2D.
    Ensures that the output at (i, j) depends only on input elements from (i, j) and earlier.

    Shape:
        Input: (N, in_channels, H, W)
        Output: (N, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super().__init__()

        # Compute padding only for top and left
        self.pad_h = (kernel_size[0] - 1) * dilation[0]
        self.pad_w = (kernel_size[1] - 1) * dilation[1]

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.pad_h, self.pad_w),
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x[:, :, : -self.pad_h, : -self.pad_w]  # Remove future pixels
