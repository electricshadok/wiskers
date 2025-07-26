import torch
import torch.nn as nn

import wiskers.common.modules.conv_blocks_2d as conv_2d
from wiskers.common.modules.activations import ActivationFct
from wiskers.common.modules.attentions_2d import SelfMultiheadAttention2D, SelfScaledDotProductAttention2D


class DownBlock2D(conv_2d.DownBlock2D):
    """
    2D Downward Block with convolution and time embedding.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation name.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H/2, W/2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        activation: str = "relu",
    ):
        super().__init__(in_channels, out_channels, activation)

        self.time_emb = nn.Sequential(
            ActivationFct.get(activation),
            nn.Linear(time_dim, out_channels),
        )

    def forward(self, x, te):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        out_x = super().forward(x)

        out_te = self.time_emb(te)  # (N, T) where T is time_dim
        out_te = out_te[:, :, None, None]  # (N, T, 1, 1)
        out_te = out_te.repeat(1, 1, out_x.shape[-2], out_x.shape[-1])  # [N, out_C, H/2, W/2]

        return out_x + out_te


class AttnDownBlock2D(DownBlock2D):
    """
    2D Downward Block with convolution, time embedding and channel attention

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation name.
        num_heads (int): Number of parallel attention heads.
            A value of 0 implies the use of scaled dot-product attention
            instead of multihead attention.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H/2, W/2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        activation: str = "relu",
        num_heads: int = 0,
    ):
        super().__init__(in_channels, out_channels, time_dim, activation)
        if num_heads == 0:
            self.attn = SelfScaledDotProductAttention2D(out_channels)
        else:
            self.attn = SelfMultiheadAttention2D(out_channels, num_heads)

    def forward(self, x, te):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        return self.attn(super().forward(x, te))


class UpBlock2D(conv_2d.UpBlock2D):
    """
    2D Upward Block with convolution, concatenation, and time embedding.

    Args:
        in_channels (int): Input channels.
        skip_channels (int): Number of channels from skip connections.
        out_channels (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation name.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H*2, W*2)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        activation: str = "relu",
    ):
        super().__init__(in_channels, skip_channels, out_channels, activation)
        self.time_emb = nn.Sequential(
            ActivationFct.get(activation),
            nn.Linear(time_dim, out_channels),
        )

    def concatenate(self, encoder_layer, decoder_layer):
        # Note: keep code but unecessary for the current unet implementation
        # encoder_shape = encoder_layer.size()[-2:]
        # decoder_shape = decoder_layer.size()[-2:]
        # if encoder_shape != decoder_shape:
        #    encoder_layer = TF.center_crop(encoder_layer, decoder_shape)

        x = torch.cat((encoder_layer, decoder_layer), dim=1)
        return x

    def forward(self, x, skip_x, te):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H*2, W*2]
        out_x = super().forward(x, skip_x)

        out_te = self.time_emb(te)  # (N, T) where T is time_dim
        out_te = out_te[:, :, None, None]  # (N, T, 1, 1)
        out_te = out_te.repeat(1, 1, out_x.shape[-2], out_x.shape[-1])  # [N, out_C, H*2, W*2]

        return out_x + out_te


class AttnUpBlock2D(UpBlock2D):
    """
    2D Upward Block with convolution, concatenation, time embedding and channel attention

    Args:
        in_channels (int): Input channels.
        skip_channels (int): Number of channels from skip connections.
        out_channels (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation name.
        num_heads (int): Number of parallel attention heads.
            A value of 0 implies the use of scaled dot-product attention
            instead of multihead attention.

    Shapes:
        Input: (N, in_C, H, W)
        Output: (N, out_C, H*2, W*2)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        activation: str = "relu",
        num_heads: int = 0,
    ):
        super().__init__(in_channels, skip_channels, out_channels, time_dim, activation)
        if num_heads == 0:
            self.attn = SelfScaledDotProductAttention2D(out_channels)
        else:
            self.attn = SelfMultiheadAttention2D(out_channels, num_heads)

    def forward(self, x, skip_x, te):
        # in: [N, in_C, H, W]
        # out: [N, out_C, H/2, W/2]
        return self.attn(super().forward(x, skip_x, te))
