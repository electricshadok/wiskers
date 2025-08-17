from typing import List, Tuple

import torch.nn as nn

from wiskers.common.modules.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DoubleConv2D,
    DownBlock2D,
    ResDoubleConv2D,
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
        activation (str): Activation function.

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
        activation: str = "relu",
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        # Input and Encoder (Down blocks and self-attention blocks)
        num_levels = len(attentions)
        self.input = DoubleConv2D(in_channels=in_channels, out_channels=widths[0], activation=activation)
        down_blocks = []
        for level_idx in range(num_levels):
            up_filters, low_filters = widths[level_idx], widths[level_idx + 1]
            if attentions[level_idx]:
                down_block = AttnDownBlock2D(up_filters, low_filters, activation, num_heads)
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
        activation (str): Activation function.

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
        activation: str = "relu",
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        num_levels = len(attentions)
        up_blocks = []
        for level_idx in range(num_levels):
            low_filters, up_filters = widths[level_idx], widths[level_idx + 1]
            if attentions[level_idx]:
                up_block = AttnUpBlock2D(low_filters, 0, up_filters, activation, num_heads)
            else:
                up_block = UpBlock2D(low_filters, 0, up_filters, activation)
            up_blocks.append(up_block)
        up_blocks.append(nn.Conv2d(widths[-1], out_channels, kernel_size=1))
        self.up_blocks = nn.Sequential(*up_blocks)

    def forward(self, x):
        return self.up_blocks(x)


class BottleneckAE(nn.Module):
    """
    Bottleneck module for Autoencoder.
    """

    def __init__(self, lowest_tensor_shape: Tuple[int, int, int], z_dim: int):
        super().__init__()
        self.lowest_tensor_shape = lowest_tensor_shape
        bot_channels, bot_tensor_h, bot_tensor_w = lowest_tensor_shape
        self.bot = ResDoubleConv2D(bot_channels, "sigmoid")
        hidden_dim = bot_channels * bot_tensor_h * bot_tensor_w
        self.bot_flatten = nn.Flatten(start_dim=1)  # (N, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, z_dim)
        self.from_latent = nn.Linear(z_dim, hidden_dim)

    def unflatten(self, z):
        bot_shape = (
            z.shape[0],
            self.lowest_tensor_shape[0],
            self.lowest_tensor_shape[1],
            self.lowest_tensor_shape[2],
        )
        unflatten = self.from_latent(z).view(*bot_shape)
        return unflatten

    def forward(self, x):
        bot = self.bot(x)
        bot_flatten = self.bot_flatten(bot)
        z = self.to_latent(bot_flatten)
        return z


class Autoencoder2D(nn.Module):
    """
    Autoencoder architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        z_dim (int): Bottleneck dimension for vae.
        image_size (tuple): Image size to with the model.
        activation (str): Activation function.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        z_dim: int = 64,
        image_size: int = 32,
        activation: str = "relu",
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        self.num_levels = len(attentions)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input and Encoder (Down blocks and self-attention blocks)
        self._encoder = Encoder(
            in_channels=in_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            activation=activation,
        )

        # Bottleneck
        # flatten the tensor (N, widths[-1], H / 2^(num_levels), W / 2^(num_levels))
        bot_img_size = image_size // 2**self.num_levels
        lowest_tensor_shape = (widths[-1], bot_img_size, bot_img_size)
        self._bottleneck = BottleneckAE(lowest_tensor_shape, z_dim)

        # Decoder (Up blocks and self-attention blocks)
        self._decoder = Decoder(
            out_channels=out_channels,
            num_heads=num_heads,
            widths=list(reversed(widths)),
            attentions=attentions,
            activation=activation,
        )

    def decoder(self, z):
        unflatten_z = self._bottleneck.unflatten(z)
        return self._decoder(unflatten_z)

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
        # Encoder
        encoder_x = self._encoder(x)

        # Bottleneck
        z = self._bottleneck(encoder_x)

        # Decoder
        decoder_x = self.decoder(z)
        return decoder_x
