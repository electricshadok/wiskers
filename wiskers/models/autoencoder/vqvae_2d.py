from typing import Tuple

import torch
import torch.nn as nn

from wiskers.common.latent_base import LatentModelBase
from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder


class VQ_VAE2D(nn.Module):
    """
    VAE architecture.

    Args:
        encoder (CNNEncoder): Prebuilt encoder module.
        decoder (CNNDecoder): Prebuilt decoder module.
        latent_model (LatentModelBase): Instantiated latent bottleneck (e.g. VectorQuantizer or VAE).
        latent_shape (tuple): (C, H, W) latent tensor shape produced by the encoder.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H, W]
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        decoder: CNNDecoder,
        latent_model: LatentModelBase,
        latent_shape: Tuple[int, int, int],
    ):
        super().__init__()

        self._encoder = encoder
        self._latent_shape = latent_shape
        latent_channels = self._latent_shape[0]

        if latent_model.code_dim != latent_channels:
            raise ValueError(
                "latent_model.code_dim must match encoder latent channels: "
                f"expected {latent_channels}, got {latent_model.code_dim}"
            )

        self._latent_model = latent_model

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
        Forward pass of the VQ-VAE model.

        Args:
            x (torch.FloatTensor): Input tensor of shape (N, in_C, H, W).

        Returns:
            recon_x: reconstructed image
            vq_loss: commitment / total VQ loss
            encoding_indices: flattened indices of selected codes
            torch.FloatTensor: Output tensor of shape (N, out_C, H, W).

        Shapes:
            in: [N, in_C, H, W]
            out: [N, out_C, H, W]
        """
        z_e = self._encoder(x)
        z_q_st, vq_loss, indices = self._latent_model(z_e)
        recon_x = self._decoder(z_q_st)
        return recon_x, vq_loss, indices
