from typing import List, Optional, Tuple, Union

import torch.nn as nn

from wiskers.common.blocks.conv_blocks_2d import (
    AttnDownBlock2D,
    AttnUpBlock2D,
    DoubleConv2D,
    DownBlock2D,
    UpBlock2D,
)
from wiskers.common.runtime.arg_utils import format_image_size


class CNNEncoder(nn.Module):
    """
    Convolutional encoder used across autoencoders.

    Args:
        in_channels (int): Number of input channels.
        stem_channels (Optional[int]): Channels produced by the stem projection
            before entering the first block. Defaults to the first entry in
            block_channels if not provided.
        num_heads (int): Number of self-attention heads.
        block_channels (List[int]): Filter width per level.
        block_attentions (List[bool]) : Enable attention per level.
        activation (nn.Module): Activation function.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H // 2^len(block_attentions), W // 2^len(block_attentions)]
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: Optional[int] = None,
        num_heads: int = 8,
        block_channels: List[int] = [32, 64, 128],
        block_attentions: List[bool] = [True, True, True],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(block_channels) != len(block_attentions):
            raise ValueError("len(block_channels) must equal len(block_attentions)")

        current_channels = (
            stem_channels if stem_channels is not None else block_channels[0]
        )

        # Input and Encoder (Down blocks and self-attention blocks)
        self._num_levels = len(block_attentions)
        self.input = DoubleConv2D(
            in_channels=in_channels,
            out_channels=current_channels,
            activation=activation,
        )
        down_blocks = []
        for level_idx in range(self._num_levels):
            up_filters, low_filters = (current_channels, block_channels[level_idx])
            if block_attentions[level_idx]:
                down_block = AttnDownBlock2D(
                    up_filters, low_filters, activation, num_heads
                )
            else:
                down_block = DownBlock2D(up_filters, low_filters, activation)
            down_blocks.append(down_block)
            current_channels = low_filters
        self._latent_channels = current_channels
        self.down_blocks = nn.Sequential(*down_blocks)

    def forward(self, x):
        return self.down_blocks(self.input(x))

    def get_latent_shape(
        self, image_size: Union[int, Tuple[int, int]]
    ) -> Tuple[int, int, int]:
        """Compute latent tensor shape given the input image size."""
        h, w = format_image_size(image_size)
        factor = 2**self._num_levels
        return self._latent_channels, h // factor, w // factor


class CNNDecoder(nn.Module):
    """
    Convolutional decoder used across autoencoders.

    Args:
        out_channels (int): Number of output channels.
        stem_channels (Optional[int]): Channels expected at the decoder input
            before the first up block. Defaults to the first entry in block_channels
            when not set.
        num_heads (int): Number of self-attention heads.
        block_channels (List[int]): Filter width per level.
        block_attentions (List[bool]) : Enable attention per level.
        activation (nn.Module): Activation function.

    Shapes:
        in: [N, in_C, H, W]
        out: [N, out_C, H * 2^len(block_attentions), W * 2^len(block_attentions)]
    """

    def __init__(
        self,
        out_channels: int = 3,
        stem_channels: Optional[int] = None,
        num_heads: int = 8,
        block_channels: List[int] = [128, 64, 32],
        block_attentions: List[bool] = [True, True, True],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if len(block_channels) != len(block_attentions):
            raise ValueError("len(block_channels) must equal len(block_attentions)")

        current_channels = (
            stem_channels if stem_channels is not None else block_channels[0]
        )

        up_blocks = []
        for level_idx in range(len(block_attentions)):
            low_filters, up_filters = (current_channels, block_channels[level_idx])
            if block_attentions[level_idx]:
                up_block = AttnUpBlock2D(
                    low_filters, 0, up_filters, activation, num_heads
                )
            else:
                up_block = UpBlock2D(low_filters, 0, up_filters, activation)
            up_blocks.append(up_block)
            current_channels = up_filters
        up_blocks.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        self.up_blocks = nn.Sequential(*up_blocks)

    def forward(self, x):
        return self.up_blocks(x)
