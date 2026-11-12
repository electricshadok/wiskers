from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1D Gated Convolutions
# ---------------------------------------------------------------------------


class GatedConv1d(nn.Module):
    """
    Implementation of a Gated Convolution in 1D.
    Uses a gating mechanism to dynamically control information flow, improving feature selection.
    Paper: "Free-Form Image Inpainting with Gated Convolution", 2019

    Shape:
        Input: (N, in_channels, L)
        Output: (N, out_channels, L_out), where:
            - L_out = (L + 2P - D(K-1) -1) / S + 1
            - P = padding, K = kernel size, D = dilation, S = stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.gate_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate


class GatedConvTranspose1d(nn.Module):
    """
    Implementation of a Gated Transposed Convolution in 1D.

    Shape:
        Input: (N, in_channels, L)
        Output: (N, out_channels, L_out), where:
            - L_out = (L - 1) * S - 2P + D(K - 1) + OP + 1
            - P = padding, K = kernel size, D = dilation, S = stride, OP = output_padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        self.gate_conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate


# ---------------------------------------------------------------------------
# 2D Gated Convolutions
# ---------------------------------------------------------------------------


class GatedConv2d(nn.Module):
    """
    Implementation of a Gated Convolution in 2D.
    Uses a gating mechanism to dynamically control information flow, improving feature selection.
    Paper: "Free-Form Image Inpainting with Gated Convolution", 2019

    Shape:
        Input: (N, in_channels, H, W)
        Output: (N, out_channels, H_out, W_out), where:
            - H_out = (H + 2P - D(K-1) -1) / S + 1
            - W_out = (W + 2P - D(K-1) -1) / S + 1
            - P = padding, K = kernel size, D = dilation, S = stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate


class GatedConvTranspose2d(nn.Module):
    """
    Implementation of a Gated Transposed Convolution in 2D.

    Shape:
        Input: (N, in_channels, H, W)
        Output: (N, out_channels, H_out, W_out), where:
            - H_out = (H - 1) * S - 2P + D(K - 1) + OP + 1
            - W_out = (W - 1) * S - 2P + D(K - 1) + OP + 1
            - P = padding, K = kernel size, D = dilation, S = stride, OP = output_padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        output_padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        self.gate_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate


# ---------------------------------------------------------------------------
# 3D Gated Convolutions
# ---------------------------------------------------------------------------


class GatedConv3d(nn.Module):
    """
    Implementation of a Gated Convolution in 3D.
    Uses a gating mechanism to dynamically control information flow, improving feature selection.
    Paper: "Free-Form Image Inpainting with Gated Convolution", 2019

    Shape:
        Input: (N, in_channels, D, H, W)
        Output: (N, out_channels, D_out, H_out, W_out), where:
            - D_out = (D + 2P - D(K-1) -1) / S + 1
            - H_out = (H + 2P - D(K-1) -1) / S + 1
            - W_out = (W + 2P - D(K-1) -1) / S + 1
            - P = padding, K = kernel size, D = dilation, S = stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        self.gate_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate


class GatedConvTranspose3d(nn.Module):
    """
    Implementation of a Gated Transposed Convolution in 3D.

    Shape:
        Input: (N, in_channels, D, H, W)
        Output: (N, out_channels, D_out, H_out, W_out), where:
            - D_out = (D - 1) * S - 2P + D(K - 1) + OP + 1
            - H_out = (H - 1) * S - 2P + D(K - 1) + OP + 1
            - W_out = (W - 1) * S - 2P + D(K - 1) + OP + 1
            - P = padding, K = kernel size, D = dilation, S = stride, OP = output_padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        output_padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.activation = activation
        self.feature_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        self.gate_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation=dilation,
        )
        nn.init.constant_(self.gate_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.activation(self.feature_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        return feature_map * gate

