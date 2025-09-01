from typing import List, Tuple, Union

import torch
import torch.nn as nn

from wiskers.autoencoder.models.ae_2d import Decoder, Encoder
from wiskers.common.arg_utils import format_image_size
from wiskers.common.modules.conv_blocks_2d import ResDoubleConv2D


class BottleneckVAE(nn.Module):
    """
    Convolutional bottleneck for Variational Autoencoder (VAE) without flattening.

    Shapes:
        in: [N, C, H, W]
        out: [N, C, H, W]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.encoder_conv = ResDoubleConv2D(channels, nn.Sigmoid())

        # Reparametrize using convolution instead of fully connected layers
        self.conv_mu = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(channels, channels, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder_conv(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
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

        self._bottleneck = BottleneckVAE(widths[-1])

        self._decoder = Decoder(
            out_channels=out_channels,
            num_heads=num_heads,
            widths=list(reversed(widths)),
            attentions=attentions,
            activation=activation,
        )

    def get_latent_shape(self):
        # downsampled 2^num_levels times in each dimension
        mid_h = self.image_size[0] // (2**self.num_levels)
        mid_w = self.image_size[1] // (2**self.num_levels)
        mid_c = self.widths[-1]
        return mid_c, mid_h, mid_w

    def decoder(self, z):
        mid_c, mid_h, mid_w = self.get_latent_shape()
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
        z, mu, logvar = self._bottleneck(encoder_x)
        decoder_x = self.decoder(z)
        return decoder_x, mu, logvar
