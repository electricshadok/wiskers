from typing import List, Tuple, Union

import torch
import torch.nn as nn

from wiskers.autoencoder.models.ae_2d import Decoder, Encoder
from wiskers.autoencoder.utils import format_image_size
from wiskers.common.modules.conv_blocks_2d import ResDoubleConv2D


class BottleneckVAE(nn.Module):
    """
    Bottleneck module for Variational Autoencoder (VAE).
    """

    def __init__(self, lowest_tensor_shape: Tuple[int, int, int], z_dim: int):
        super().__init__()
        self.lowest_tensor_shape = lowest_tensor_shape
        bot_channels, bot_tensor_h, bot_tensor_w = lowest_tensor_shape
        self.bot = ResDoubleConv2D(bot_channels, nn.Sigmoid())
        hidden_dim = bot_channels * bot_tensor_h * bot_tensor_w
        self.bot_flatten = nn.Flatten(start_dim=1)  # (N, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.bot_unflatten = nn.Linear(z_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def unflatten(self, z):
        bot_shape = (
            z.shape[0],
            self.lowest_tensor_shape[0],
            self.lowest_tensor_shape[1],
            self.lowest_tensor_shape[2],
        )
        unflatten = self.bot_unflatten(z).view(*bot_shape)
        return unflatten

    def forward(self, x):
        bot = self.bot(x)
        bot_flatten = self.bot_flatten(bot)
        mu = self.fc_mu(bot_flatten)
        logvar = self.fc_logvar(bot_flatten)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAE2D(nn.Module):
    """
    VAE architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        z_dim (int): Bottleneck dimension for vae.
        image_size (int or tuple): Input image size (H, W).
        activation (nn.Module): Activation function.

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
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        self.num_levels = len(attentions)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convert image_size to (H, W)
        image_size = format_image_size(image_size)

        # Input and Encoder (Down blocks and self-attention blocks)
        self._encoder = Encoder(
            in_channels=in_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            activation=activation,
        )

        # Bottleneck
        # Image is downsampled 2^num_levels times in each dimension
        h_bot = image_size[0] // (2**self.num_levels)
        w_bot = image_size[1] // (2**self.num_levels)
        lowest_tensor_shape = (widths[-1], h_bot, w_bot)
        self._bottleneck = BottleneckVAE(lowest_tensor_shape, z_dim)

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
        z, mu, logvar = self._bottleneck(encoder_x)

        # Decoder
        decoder_x = self.decoder(z)
        return decoder_x, mu, logvar
