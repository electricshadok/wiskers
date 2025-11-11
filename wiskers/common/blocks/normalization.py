import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization
    Paper: "Arbitrary style transfer in realtime with adaptive instance normalization", 2017.
    AdaIN doesn't have parameters to learn.

    Shapes:
        Input: (N,C,H,W)
        Output: (N,C,H,W)
    """

    EPSILON = 1e-5

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Harmonize the mean and standard deviation of the content (x) with the style (y)"""
        # compute mean and std for content (x) and style (y)
        x_mean, x_var = torch.var_mean(x, dim=[2, 3], keepdim=True)
        y_mean, y_var = torch.var_mean(y, dim=[2, 3], keepdim=True)

        # Standardize content features
        normalized_x = (x - x_mean) / torch.sqrt(x_var + self.EPSILON)

        # Scale and shift the normalized content to match style's statistics
        stylized_x = normalized_x * torch.sqrt(y_var + self.EPSILON) + y_mean

        return stylized_x


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT)
    Paper: Transformers without Normalization
    GitHub(Implementation): https://jiachenzhu.github.io/DyT/

    Shapes:
        Input: (N, C) or (N, C, H, W) depending on `channels_last`
        Output: (N, C) or (N, C, H, W)

    Args:
        normalized_shape (int): The number of features (C) to normalize.
        channels_last (bool): If True, applies element-wise transformation.
                              If False, applies channel-wise transformation for spatial inputs.
        alpha_init_value (float): Initial value for alpha parameter.

    """

    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            # Element-wise transformation
            x = x * self.weight + self.bias
        else:
            # Broadcast across spatial dims
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
