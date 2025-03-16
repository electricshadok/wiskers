import torch.nn as nn


class CausalConv1d(nn.Module):
    """
    Implementation of a causal convolution in 1D
    Ensures that the output at time t depends only on input elements from time t and earlier.

    Shape:
        Input: (N, in_channels, seq_length)
        Output (N, out_channels, seq_length)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv1d(x)
        return x[:, :, : -self.padding]  # Remove future pixels
