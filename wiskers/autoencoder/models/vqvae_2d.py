from typing import List, Tuple, Union

import torch
import torch.nn as nn

from wiskers.autoencoder.models.ae_2d import Decoder, Encoder
from wiskers.common.arg_utils import format_image_size
from wiskers.common.blocks.quantizer import VectorQuantizer


class VQ_VAE2D(nn.Module):
    """
    VAE architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        image_size (int or tuple): Input image size (H, W).
        activation (nn.Module): Activation function.
        num_codes (int): Number of discrete embeddings in the codebook (K).
        code_dim (int): Dimensionality of each embedding vector (D).
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
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: nn.Module = nn.ReLU(),
        num_codes: int = 512,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
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

        self._quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=widths[-1],
            beta=beta,
            use_ema=use_ema,
            decay=decay,
            eps=eps,
        )

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
