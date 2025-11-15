from typing import List

import torch.nn as nn

from wiskers.common.blocks.conv_blocks_2d import (
    DoubleConv2D,
    ResDoubleConv2D,
)
from wiskers.common.blocks.positional_encoding import SinusoidalPositionEmbedding
from wiskers.models.diffusion.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)


class UNet2D(nn.Module):
    """
    U-Net architecture for diffusion models.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_dim (int): Size of the time dimension.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        activation (nn.Module): Activation function.

    Shapes:
        in: [N, 3, 32, 32]
        out: [N, 3, 32, 32]

    Example internal shape with input (N, 32, 32, 32)
        # Input and positional embedding
        input (N, 32, 32, 32)
        te = (B, time_dim)
        # Encoder
        level_0: (N, 64, 16, 16)
        level_1: (N, 128, 8, 8)
        level_2: (N, 256, 4, 4)
        # Bottleneck
        (N, 256, 4, 4)
        # Decoder
        level_2: (N, 128, 8, 8)
        level_1: (N, 64, 16, 16)
        level_0: (N, 32, 32, 32)
        # Output
            (N, 3, 32, 32)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(widths) - 1 != len(attentions):
            raise ValueError("Wrong input len(widths)-1 != len(attentions)")

        self.num_levels = len(attentions)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim

        # Positional embedding
        self.pe = SinusoidalPositionEmbedding(time_dim)

        # Input
        self.input = DoubleConv2D(
            in_channels=in_channels, out_channels=widths[0], activation=activation
        )

        # Encoder (Down blocks and self-attention blocks)
        self.down_blocks = []
        for level_idx in range(self.num_levels):
            up_filters, low_filters = widths[level_idx], widths[level_idx + 1]
            if attentions[level_idx]:
                down_block = AttnDownBlock2D(
                    up_filters, low_filters, time_dim, activation, num_heads
                )
            else:
                down_block = DownBlock2D(up_filters, low_filters, time_dim, activation)
            self.down_blocks.append(down_block)
        self.down_blocks = nn.ModuleList(self.down_blocks)

        # Bottleneck
        self.bot = ResDoubleConv2D(widths[-1], activation)

        # Decoder (Up blocks and self-attention blocks)
        self.up_blocks = []
        for level_idx in reversed(range(self.num_levels)):
            low_filters, up_filters = widths[level_idx + 1], widths[level_idx]
            if attentions[level_idx]:
                up_block = AttnUpBlock2D(
                    low_filters, up_filters, up_filters, time_dim, activation, num_heads
                )
            else:
                up_block = UpBlock2D(
                    low_filters, up_filters, up_filters, time_dim, activation
                )
            self.up_blocks.append(up_block)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # Output
        self.pointwise = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Forward pass of the U-Net diffusion model.

        Args:
            x (torch.FloatTensor): Input tensor of shape (N, in_C, H, W).
            t (torch.LongTensor): Time tensor of shape (N)

        Returns:
            torch.FloatTensor: Output tensor of shape (N, out_C, H, W).

        Shape:
            in: [N, in_C, H, W]
            out: [N, out_C, H, W]
        """
        skip_x = []  # skip connections
        in_x = self.input(x)
        skip_x.append(in_x)
        te = self.pe(t)

        # Encoder
        for down_block in self.down_blocks:
            x_prev = in_x if len(skip_x) == 0 else skip_x[-1]
            x_n = down_block(x_prev, te)
            skip_x.append(x_n)

        # Bottleneck
        bot = self.bot(skip_x.pop())

        # Decoder
        x_n = bot
        for up_block in self.up_blocks:
            x_n = up_block(x_n, skip_x.pop(), te)

        # Output
        out = self.pointwise(x_n)
        return out
