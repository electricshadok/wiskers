from typing import Tuple, Union

import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Implementation of a causal convolution in 1D.
    Ensures that the output at time t depends only on input elements from time t and earlier.

    Shape:
        Input: (N, in_channels, seq_length)
        Output: (N, out_channels, seq_length)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv1d(x)
        return x[:, :, : -self.padding]  # Remove future pixels


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
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (1, 1),
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        bias: bool = False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

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
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        bias: bool = False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Compute padding only for front (depth), top (height), and left (width)
        self.pad_d = (kernel_size[0] - 1) * dilation[0]  # Temporal padding
        self.pad_h = (kernel_size[1] - 1) * dilation[1]  # Height padding
        self.pad_w = (kernel_size[2] - 1) * dilation[2]  # Width padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(self.pad_d, self.pad_h, self.pad_w),
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x[:, :, : -self.pad_d, : -self.pad_h, : -self.pad_w]  # Remove future values

