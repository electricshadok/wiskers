from typing import List, Tuple, Union

import torch.nn as nn

from wiskers.common.arg_utils import format_image_size
from wiskers.common.modules.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DoubleConv2D,
    DownBlock2D,
    UpBlock2D,
)


class Encoder(nn.Module):
    """
    Encoder for VAE

    Args:
        in_channels (int): Number of input channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        activation (nn.Module): Activation function.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H // 2^len(attentions), W // 2^len(attentions)]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        # Input and Encoder (Down blocks and self-attention blocks)
        num_levels = len(attentions)
        self.input = DoubleConv2D(
            in_channels=in_channels, out_channels=widths[0], activation=activation
        )
        down_blocks = []
        for level_idx in range(num_levels):
            up_filters, low_filters = widths[level_idx], widths[level_idx + 1]
            if attentions[level_idx]:
                down_block = AttnDownBlock2D(
                    up_filters, low_filters, activation, num_heads
                )
            else:
                down_block = DownBlock2D(up_filters, low_filters, activation)
            down_blocks.append(down_block)
        self.down_blocks = nn.Sequential(*down_blocks)

    def forward(self, x):
        return self.down_blocks(self.input(x))


class Decoder(nn.Module):
    """
    Decoder for VAE

    Args:
        in_channels (int): Number of input channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        activation (nn.Module): Activation function.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H * 2^len(attentions), W * 2^len(attentions)]
    """

    def __init__(
        self,
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [256, 128, 64, 32],
        attentions: List[bool] = [True, True, True],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        num_levels = len(attentions)
        up_blocks = []
        for level_idx in range(num_levels):
            low_filters, up_filters = widths[level_idx], widths[level_idx + 1]
            if attentions[level_idx]:
                up_block = AttnUpBlock2D(
                    low_filters, 0, up_filters, activation, num_heads
                )
            else:
                up_block = UpBlock2D(low_filters, 0, up_filters, activation)
            up_blocks.append(up_block)
        up_blocks.append(nn.Conv2d(widths[-1], out_channels, kernel_size=1))
        self.up_blocks = nn.Sequential(*up_blocks)

    def forward(self, x):
        return self.up_blocks(x)


class Autoencoder2D(nn.Module):
    """
    Autoencoder architecture with a fully convolutional bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]): Enable attention per level.
        image_size (int or tuple): Input image size (H, W).
        activation (nn.Module): Activation function.

    Shapes:
        Input: [N, in_C, H, W]
        Output: [N, out_C, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        self.num_levels = len(attentions)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.widths = widths
        self.image_size = format_image_size(image_size)

        self._encoder = Encoder(
            in_channels=in_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            activation=activation,
        )

        self._mid_block = DoubleConv2D(
            in_channels=widths[-1], out_channels=widths[-1], activation=nn.ReLU()
        )

        self._decoder = Decoder(
            out_channels=out_channels,
            num_heads=num_heads,
            widths=list(reversed(widths)),
            attentions=attentions,
            activation=activation,
        )

    def get_expected_shape(self):
        # downsampled 2^num_levels times in each dimension
        mid_h = self.image_size[0] // (2**self.num_levels)
        mid_w = self.image_size[1] // (2**self.num_levels)
        mid_c = self.widths[-1]
        return mid_c, mid_h, mid_w

    def decoder(self, z):
        mid_c, mid_h, mid_w = self.get_expected_shape()
        expected_shape = (z.shape[0], mid_c, mid_h, mid_w)
        if z.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Expected latent shape {expected_shape}, but got {z.shape}"
            )
        return self._decoder(z)

    def forward(self, x):
        """
        Forward pass of the U-Net diffusion model.

        Args:
            x (torch.FloatTensor): Input tensor of shape (N, in_C, H, W).

        Returns:
            torch.FloatTensor: Output tensor of shape (N, out_C, H, W).

        Shapes:
            in: [N, in_C, H, W]
            out: [N, out_C, H, W]
        """
        encoder_x = self._encoder(x)
        z = self._mid_block(encoder_x)
        decoder_x = self.decoder(z)
        return decoder_x
