import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels: int, mode: str = "embedded_gaussian"):
        """
        Implementation of Non-local Neural Networks
        Paper: https://arxiv.org/pdf/1711.07971.pdf

        Shapes:
            Input: (N, C, T, H, W)
            Output: (N, C, T, H, W)
        """
        super().__init__()

        expected_modes = ["embedded_gaussian", "gaussian", "dot_product", "concatenated"]
        if mode not in expected_modes:
            raise ValueError(f"{mode} is not supported. Expected {expected_modes}")

        if mode != "embedded_gaussian":
            # TODO - for now only implemented the embedded gaussian mode
            raise NotImplementedError(f"{mode} is not implemented")

        self.hidden_channels = max(in_channels // 2, 1)
        self.in_channels = in_channels
        self.mode = mode

        self.g = nn.Conv3d(in_channels, self.hidden_channels, kernel_size=1)
        self.theta = nn.Conv3d(in_channels, self.hidden_channels, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, self.hidden_channels, kernel_size=1)
        self.W_z = nn.Conv3d(self.hidden_channels, self.in_channels, kernel_size=1)

    def forward(self, x):
        N, C, T, H, W = x.size()

        g_x = self.g(x)  # (N, C, T, H, W)
        g_x = g_x.view(N, self.hidden_channels, -1)  # (N, C, THW)
        g_x = g_x.permute(0, 2, 1)  # (N, THW, C)

        theta_x = self.theta(x)  # (N, C, T, H, W)
        theta_x = theta_x.view(N, self.hidden_channels, -1)  # (N, C, THW)
        theta_x = theta_x.permute(0, 2, 1)  # (N, THW, C)

        phi_x = self.phi(x)  # (N, C, T, H, W)
        phi_x = phi_x.view(N, self.hidden_channels, -1)  # (N, C, THW)

        f = torch.matmul(theta_x, phi_x)  # (N, THW, THW)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)  # (N, THW, C)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, THW)
        y = y.view(N, self.hidden_channels, T, H, W)

        return self.W_z(y) + x
