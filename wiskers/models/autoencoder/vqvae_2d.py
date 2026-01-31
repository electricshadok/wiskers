from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from wiskers.common.blocks.quantizer import VectorQuantizer
from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder


class VQ_VAE2D(nn.Module):
    """
    VAE architecture.

    Args:
        in_channels (int): Number of input channels.
        stem_channels (Optional[int]): Channels produced by the encoder stem
            before entering the first down block.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        block_channels (List[int]): Filter width per level.
        block_attentions (List[bool]) : Enable attention per level.
        image_size (int or tuple): Input image size (H, W).
        activation (nn.Module): Activation function.
        num_codes (int): Number of discrete embeddings in the codebook (K).
        beta (float): Weight for the commitment loss term, typically between 0.1 and 0.5.
        use_ema (bool): Whether to use EMA updates for the codebook.
        decay (float): EMA decay factor (only used if use_ema=True).
        eps (float): Small constant for numerical stability.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: Optional[int] = None,
        out_channels: int = 3,
        num_heads: int = 8,
        block_channels: List[int] = [32, 64, 128],
        block_attentions: List[bool] = [True, True, True],
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: nn.Module = nn.ReLU(),
        num_codes: int = 512,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()

        self._encoder = CNNEncoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            num_heads=num_heads,
            block_channels=block_channels,
            block_attentions=block_attentions,
            activation=activation,
        )
        self._latent_shape = self._encoder.get_latent_shape(image_size)

        self._quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=block_channels[-1],
            beta=beta,
            use_ema=use_ema,
            decay=decay,
            eps=eps,
        )

        self._decoder = CNNDecoder(
            out_channels=out_channels,
            num_heads=num_heads,
            block_channels=list(reversed(block_channels)),
            block_attentions=block_attentions,
            activation=activation,
        )

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
        Forward pass of the U-Net diffusion model.

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
        z_q_st, vq_loss, indices = self._quantizer(z_e)
        recon_x = self._decoder(z_q_st)
        return recon_x, vq_loss, indices
