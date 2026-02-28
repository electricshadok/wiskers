import torch.nn as nn


class LatentModelBase(nn.Module):
    """
    Base class for latent bottleneck models (e.g., VectorQuantizer, VAE).
    Attributes:
        code_dim (int): Dimensionality of the latent code channels.
    """

    def __init__(self, code_dim: int):
        super().__init__()
        self.code_dim = code_dim
