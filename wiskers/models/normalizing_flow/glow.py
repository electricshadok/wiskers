import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleConv1x1(nn.Module):
    """
    Implementation of an invertible 1x1 convolutional layer.
    Paper: Glow: Generative Flow with Invertible 1x1 Convolutions
    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, num_channels: int):
        super().__init__()

        self.num_channels = num_channels

        # Initialize weights to be orthogonal
        self.w = nn.Parameter(torch.Tensor(num_channels, num_channels))
        nn.init.orthogonal_(self.w)

        # Another way to initialize w to be orthogonal is to use QR decomposition:
        # w_init = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        # self.w = nn.Parameter(w_init)

    def forward(self, x, logdet=None, reverse=False):
        """
        Computes 1x1 convolution and updates the log-determinant.
        In flow, the encoder and decoder is the same hence the reverse flag.
        If reverse is False, we apply W; if True, we apply W^-1.

        Args:
            x: Input tensor (N, C, H, W).
            logdet: Running log-likelihood contribution. Tracking this is
                   required by the Change of Variables formula to ensure the
                   total probability distribution remains normalized (see paper).
            reverse: If True, performs the inverse operation (W^-1).
        """
        batch, c, h, w = x.size()

        if logdet is None:
            logdet = torch.zeros(batch, device=x.device)

        # Equation 9: log-det = h * w * log|det(W)|
        # This accounts for the volume change across all pixels
        _, logabsdet = torch.linalg.slogdet(self.w)
        dlogdet = h * w * logabsdet

        if not reverse:
            # Forward: z = x * W
            # Reshape [C, C] to [C, C, 1, 1] for the conv2d engine
            conv_w = self.w.view(c, c, 1, 1)
            z = F.conv2d(x, conv_w)
            logdet += dlogdet
        else:
            # Inverse: x = z * W^-1
            w_inv = torch.inverse(self.w)
            z = F.conv2d(x, w_inv.view(c, c, 1, 1))
            logdet -= dlogdet

        return z, logdet


class ActNorm(nn.Module):
    """
    Activation Normalization (ActNorm) layer as used in Glow.
    Initializes scale and bias per-channel using the first minibatch so that
    the activations per-channel have zero mean and unit variance.

    Shapes:
        Input: (N, C, H, W)
        Output: (N, C, H, W)
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        # bias and logs are registered parameters so they move to correct device
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        # flag to indicate whether parameters were initialized from data
        self.initialized = False

    def initialize_parameters(self, x: torch.Tensor):
        # compute mean and std over batch and spatial dims
        with torch.no_grad():
            N, C, H, W = x.shape
            assert C == self.num_channels
            # mean over N,H,W -> shape (C,)
            mean = x.permute(1, 0, 2, 3).contiguous().view(C, -1).mean(dim=1)
            var = (
                x.permute(1, 0, 2, 3)
                .contiguous()
                .view(C, -1)
                .var(dim=1, unbiased=False)
            )
            std = torch.sqrt(var + self.eps)

            # set bias so that output mean becomes 0: x + bias => mean 0 => bias = -mean
            self.bias.data = (-mean).view(1, C, 1, 1)

            # set logs so that output std becomes 1: (x + bias) * exp(logs) => std 1
            # exp(logs) = 1/std  => logs = -log(std)
            self.logs.data = (-torch.log(std)).view(1, C, 1, 1)

            self.initialized = True

    def forward(
        self, x: torch.Tensor, logdet: torch.Tensor = None, reverse: bool = False
    ):
        N, C, H, W = x.shape

        if not self.initialized and not reverse:
            # initialize from the first batch seen in forward pass
            self.initialize_parameters(x)

        if logdet is None:
            logdet = torch.zeros(x.size(0), device=x.device)

        # scale is positive scalar per channel
        scale = torch.exp(self.logs)
        if not reverse:
            # y = (x + bias) * scale
            y = (x + self.bias) * scale
            # change in log-det: H*W*sum(log|scale|) for each sample
            dlogdet = H * W * torch.sum(self.logs)
            logdet = logdet + dlogdet
            return y, logdet
        else:
            # inverse: x = y / scale - bias
            x_rec = x / scale - self.bias
            dlogdet = H * W * torch.sum(self.logs)
            logdet = logdet - dlogdet
            return x_rec, logdet
