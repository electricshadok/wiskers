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
