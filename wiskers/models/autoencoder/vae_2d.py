from typing import Tuple, Union

import torch
import torch.nn as nn

from wiskers.common.blocks.conv_blocks_2d import ResDoubleConv2D
from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder


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
        encoder (CNNEncoder): Prebuilt encoder module.
        decoder (CNNDecoder): Prebuilt decoder module.
        image_size (int or tuple): Input image size (H, W).

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H, W]
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        decoder: CNNDecoder,
        image_size: Union[int, Tuple[int, int]] = 32,
    ):
        super().__init__()

        self._encoder = encoder
        self._latent_shape = self._encoder.get_latent_shape(image_size)
        latent_channels = self._latent_shape[0]

        self._bottleneck = BottleneckVAE(latent_channels)

        self._decoder = decoder

    def get_latent_shape(self):
        # downsampled 2^num_levels times in each dimension
        return self._latent_shape

    def decoder(self, z):
        if not torch.jit.is_tracing():
            # This block ensures certain operations (like shape assertions or debug logging)
            # are only executed during eager mode (normal Python execution).
            # During JIT tracing, Python control flow depending on tensor values cannot be
            # captured correctly, so we avoid tracing potentially problematic code paths.
            mid_c, mid_h, mid_w = self.get_latent_shape()
            torch._assert(
                z.size(1) == mid_c and z.size(2) == mid_h and z.size(3) == mid_w,
                "Shape mismatch",
            )
        return self._decoder(z)

    def forward(self, x):
        """
        Forward pass of the VAE model.

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
