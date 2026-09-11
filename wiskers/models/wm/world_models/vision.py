from typing import List, Tuple

import torch
import torch.nn as nn

from wiskers.models.gen.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder


class Vision(nn.Module):
    """
    Vision model (V) from Ha & Schmidhuber, 2018.
    "World Models" — https://arxiv.org/abs/1803.10122

    A convolutional VAE that maps raw frames to a compact *flat* latent vector
    z ∈ R^latent_size. Unlike VAE2D (which keeps a spatial latent map), V
    produces a 1-D latent vector, matching the paper's original formulation.

    Architecture:
        Encoder   : CNNEncoder  →  [N, C', H', W']
        Flatten                 →  [N, C'·H'·W']
        fc_mu     : Linear      →  mu      [N, latent_size]
        fc_logvar : Linear      →  logvar  [N, latent_size]
        z = reparameterize(mu, logvar)     [N, latent_size]
        fc_decode : Linear      →  [N, C'·H'·W']
        Unflatten               →  [N, C', H', W']
        Decoder   : CNNDecoder  →  x_hat   [N, in_channels, H, W]

    Training stages (World Models 2018):
        Stage 1 — train V end-to-end with reconstruction + KL loss  ← this module
        Stage 2 — freeze V; feed encode_deterministic(x) to MDN-RNN
        Stage 3 — freeze V + M; optimise Controller with CMA-ES

    Args:
        image_size       (Tuple[int, int]): Input spatial size (H, W). Default: (64, 64).
        in_channels      (int): Number of input channels. Default: 3.
        latent_size      (int): Dimensionality of the flat latent vector z. Default: 32.
        block_channels   (List[int]): Filter widths per encoder level.
            The paper uses a simple 4-level CNN; no attention.
        block_attentions (List[bool]): Attention flags per level. All False = plain CNN.
        num_heads        (int): Attention heads (only used when attention is enabled).
        activation       (nn.Module): Activation function shared by encoder and decoder.

    Shapes:
        encode(x)               [N, C, H, W] → (z, mu, logvar)  each [N, latent_size]
        encode_deterministic(x) [N, C, H, W] → mu [N, latent_size]   (no noise)
        decode(z)               [N, latent_size] → x_hat [N, C, H, W]  ∈ [0, 1]
        forward(x)              [N, C, H, W] → (x_hat, mu, logvar)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        in_channels: int = 3,
        latent_size: int = 32,
        # Paper uses a plain 4-level CNN without attention
        block_channels: List[int] = [32, 64, 128, 256],
        block_attentions: List[bool] = [False, False, False, False],
        num_heads: int = 8,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_size = latent_size

        # ------------------------------------------------------------------ #
        #  Encoder backbone — shared with the gen library                     #
        # ------------------------------------------------------------------ #
        self._encoder = CNNEncoder(
            in_channels=in_channels,
            block_channels=block_channels,
            block_attentions=block_attentions,
            num_heads=num_heads,
            activation=activation,
        )

        # Spatial shape produced by the encoder: (C', H', W')
        enc_c, enc_h, enc_w = self._encoder.get_latent_shape(image_size)
        self._enc_c = enc_c
        self._enc_h = enc_h
        self._enc_w = enc_w
        flat_size = enc_c * enc_h * enc_w  # e.g. 256*4*4 = 4096 for 64x64

        # ------------------------------------------------------------------ #
        #  Flat latent heads                                                   #
        # ------------------------------------------------------------------ #
        self._flatten = nn.Flatten()
        self._fc_mu = nn.Linear(flat_size, latent_size)
        self._fc_logvar = nn.Linear(flat_size, latent_size)

        # ------------------------------------------------------------------ #
        #  Decoder projection: z → spatial map before CNN decoder             #
        # ------------------------------------------------------------------ #
        self._fc_decode = nn.Linear(latent_size, flat_size)
        self._unflatten = nn.Unflatten(1, (enc_c, enc_h, enc_w))

        # ------------------------------------------------------------------ #
        #  Decoder backbone — shared with the gen library                     #
        # ------------------------------------------------------------------ #
        self._decoder = CNNDecoder(
            out_channels=in_channels,
            block_channels=list(reversed(block_channels)),
            block_attentions=block_attentions,
            num_heads=num_heads,
            activation=activation,
        )

    # ---------------------------------------------------------------------- #
    #  Core methods                                                            #
    # ---------------------------------------------------------------------- #

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps*std,  eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode raw observations into the latent distribution.

        Args:
            x (torch.Tensor): Input frames [N, C, H, W].

        Returns:
            z      (torch.Tensor): Sampled latent via reparameterization [N, latent_size].
            mu     (torch.Tensor): Distribution mean   [N, latent_size].
            logvar (torch.Tensor): Log-variance        [N, latent_size].
        """
        h = self._encoder(x)          # [N, C', H', W']
        h = self._flatten(h)          # [N, C'*H'*W']
        mu = self._fc_mu(h)           # [N, latent_size]
        logvar = self._fc_logvar(h)   # [N, latent_size]
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic encode — returns mu only (no noise added).

        Used at Stage 2 to produce stable latent codes z_t fed to the MDN-RNN.
        Decorated with @no_grad because V is frozen during Stage 2.

        Args:
            x (torch.Tensor): Input frames [N, C, H, W].

        Returns:
            torch.Tensor: Latent mean mu [N, latent_size].
        """
        h = self._encoder(x)
        h = self._flatten(h)
        return self._fc_mu(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a flat latent vector back to observation space.

        Args:
            z (torch.Tensor): Latent vector [N, latent_size].

        Returns:
            torch.Tensor: Reconstructed frame [N, C, H, W] with values in [0, 1].
        """
        h = self._fc_decode(z)    # [N, C'*H'*W']
        h = self._unflatten(h)    # [N, C', H', W']
        x_hat = self._decoder(h)  # [N, C, H, W]
        return torch.sigmoid(x_hat)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass. Used during Stage 1 training.

        Loss = reconstruction_loss(x, x_hat) + beta * KL(q(z|x) || p(z))
        where KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        Args:
            x (torch.Tensor): Input frames [N, C, H, W].

        Returns:
            x_hat  (torch.Tensor): Reconstructed frame [N, C, H, W].
            mu     (torch.Tensor): Distribution mean   [N, latent_size].
            logvar (torch.Tensor): Log-variance        [N, latent_size].
        """
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
