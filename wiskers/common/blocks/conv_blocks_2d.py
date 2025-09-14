from typing import Optional

import torch  # noqa: F401
import torch.nn as nn

from wiskers.common.blocks.attentions_2d import (
    SelfMultiheadAttention2D,
    SelfScaledDotProductAttention2D,
)  # noqa: F401


class DoubleConv2D(nn.Module):
    """
    Double convolutional block.

    Args:
        in_channels (int): Input channel count.
        mid_channels (Optional[int]): Mid channel count. if None use (in_channels + out_channels) // 2
        out_channels (int): Output channel count.
        activation (nn.Module): Activation function.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: Optional[int] = None,
        out_channels: int = 3,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = (in_channels + out_channels) // 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),  # layer normalization
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),  # layer normalization
        )

    def forward(self, x):
        # in: [N, in_C, H, W]
        # out: [N, out_C, W, H]
        return self.net(x)


class ResDoubleConv2D(DoubleConv2D):
    """
    Double convolutional block with a residual connection.

    Args:
        channels (int): Input and output channel count.
        activation (nn.Module): Activation function.

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__(in_channels=channels, out_channels=channels)
        self.activation = activation

    def forward(self, x):
        # in: [N, C, H, W]
        # out: [N, C, W, H]
        return self.activation(x + super().forward(x))


class DownBlock2D(nn.Module):
    """
    2D downward block for UNet

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        activation (nn.Module): Activation function.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H/2, W/2)

    Notes:
        Default nn.Conv2d is kernel_size=2 => stride=2, padding=0
        Future: user input for kernel size
            kernel_size=3, stride=2, padding=1
            kernel_size=4 => stride=2, padding=1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            DoubleConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=activation,
            ),
        )

    def forward(self, x):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        return self.conv_net(x)


class AttnDownBlock2D(DownBlock2D):
    """
    2D downward block with channel attention for UNet

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        activation (nn.Module): Activation function.
        num_heads (int): Number of parallel attention heads.
            A value of 0 implies the use of scaled dot-product attention
            instead of multihead attention.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H/2, W/2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module = nn.ReLU(),
        num_heads: int = 0,
    ):
        super().__init__(in_channels, out_channels, activation)
        if num_heads == 0:
            self.attn = SelfScaledDotProductAttention2D(out_channels)
        else:
            self.attn = SelfMultiheadAttention2D(out_channels, num_heads)

    def forward(self, x):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        return self.attn(super().forward(x))


class UpBlock2D(nn.Module):
    """
    2D upward block for UNet

    Args:
        in_channels (int): Input channels.
        skip_channels (int): Number of channels from skip connections.
        out_channels (int): Output channels.
        activation (nn.Module): Activation function.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H*2, W*2)

    Notes:
        Default nn.Conv2d is kernel_size=2, stride=2, padding=0, output_padding=0
        Future: user input for kernel size
            kernel_size=3, stride=2, padding=1, output_padding=1
            kernel_size=4, stride=2, padding=1, output_padding=0
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.net = DoubleConv2D(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            activation=activation,
        )

    def concatenate(self, encoder_layer, decoder_layer):
        # Note: keep code but unecessary for the current unet implementation
        # encoder_shape = encoder_layer.size()[-2:]
        # decoder_shape = decoder_layer.size()[-2:]
        # if encoder_shape != decoder_shape:
        #    encoder_layer = TF.center_crop(encoder_layer, decoder_shape)

        x = torch.cat((encoder_layer, decoder_layer), dim=1)
        return x

    def forward(self, x, skip_x: Optional[torch.Tensor] = None):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H*2, W*2]
        out_x = self.up(x)  # [B, out_channels, H*2, W*2]
        # Concatenate skip connection along the channels
        if skip_x is not None:
            out_x = self.concatenate(skip_x, out_x)

        return self.net(out_x)


class AttnUpBlock2D(UpBlock2D):
    """
    2D upward block with channel attention for UNet

    Args:
        in_channels (int): Input channels.
        skip_channels (int): Number of channels from skip connections.
        out_channels (int): Output channels.
        activation (nn.Module): Activation function.
        num_heads (int): Number of parallel attention heads.
            A value of 0 implies the use of scaled dot-product attention
            instead of multihead attention.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H*2, W*2)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation: nn.Module = nn.ReLU(),
        num_heads: int = 0,
    ):
        super().__init__(in_channels, skip_channels, out_channels, activation)
        if num_heads == 0:
            self.attn = SelfScaledDotProductAttention2D(out_channels)
        else:
            self.attn = SelfMultiheadAttention2D(out_channels, num_heads)

    def forward(self, x, skip_x: Optional[torch.Tensor] = None):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        return self.attn(super().forward(x, skip_x))


class SeparableConv2d(nn.Module):
    """
    Implements a separable convolution block.

    This module separates the convolution into two parts
    - The depthwise convolution performs a spatial convolution independently for each input channel
    - The pointwise convolution is a 1x1 convolution that mixes the channels

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        # Depthwise convolution
        # Each input channel is convolved independently
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )

        # Pointwise convolution
        # A 1x1 convolution to mix the channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )

    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise(x)
        # Apply pointwise convolution
        x = self.pointwise(x)
        return x
