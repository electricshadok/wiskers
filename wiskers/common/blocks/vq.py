import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    """
    A discrete embedding space used for vector quantization.
    Stores K embedding vectors of dimension D.

    Args:
        num_codes (int): Number of discrete embeddings in the codebook (K).
        code_dim (int): Dimensionality of each embedding vector (D).
    """

    def __init__(self, num_codes: int, code_dim: int):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim

        # Learnable embedding matrix (K, D)
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))

    def compute_distances(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance between z_flat and all codebook entries.

        Args:
            encoding (Tensor): Flattened input tensor of shape (N, D).

        Returns:
            distance_matrix (Tensor): (N, K) — distances to each codebook vector.
        """
        # Compute squared Euclidean distances between each latent vector and all codebook entries
        # Formula: ||z - c||² = ||z||² + ||c||² - 2·z·c^T
        z2 = (encoding**2).sum(dim=1, keepdim=True)  # (N, 1)  — squared norms of inputs
        c2 = (self.codebook**2).sum(dim=1)  # (1, K)  — squared norms of codes
        c2 = c2.unsqueeze(0)  # (1, K)
        zc = encoding @ self.codebook.t()  # (N, K)

        # Full distance matrix between inputs and codebook entries: (N, K)
        distance_matrix = z2 + c2 - 2 * zc

        return distance_matrix

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve embedding vectors by their indices.

        Args:
            indices (Tensor): (N,) — nearest code indices.

        Returns:
            Tensor: (N, D) — corresponding embedding vectors.
        """
        return self.codebook[indices]


class VectorQuantizer(nn.Module):
    """
    Implementation of the Vector Quantization (VQ) layer used in VQ-VAE models.
    This module discretizes continuous latent representations by mapping each latent
    vector to the nearest embedding vector from a learned codebook.

    Paper: "Neural Discrete Representation Learning" (van den Oord et al., 2017).

    Args:
        num_codes (int): Number of discrete embeddings in the codebook (K).
        code_dim (int): Dimensionality of each embedding vector (D).
        beta (float): Weight for the commitment loss term, typically between 0.1 and 0.5.

    Shapes:
        Input:
            - x: (B, D, H, W) — Continuous latent representation from encoder.
        Output:
            - z_q_st: (B, D, H, W) — Quantized latent representation used by decoder (with STE).
            - vq_loss: () — Scalar loss combining codebook and commitment terms.
            - encoding_indices: (B * H * W) — Flattened indices of selected codebook entries.
    """

    def __init__(self, num_codes=1024, code_dim=64, beta=0.25):
        super().__init__()
        self.num_codes = num_codes  # Number of embeddings in the codebook (K)
        self.code_dim = code_dim  # Dimensionality of each embedding vector (D)
        self.beta = beta  # Commitment loss weight

        # Learnable embedding matrix serving as the discrete codebook (K, D)
        self.codebook = Codebook(num_codes, code_dim)

    def _quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replace each latent vector with its nearest codebook vector.

        Args:
            z (Tensor): Encoder feature map of shape (B, D, H, W).

        Returns:
            z_q (Tensor): Quantized feature map of shape (B, D, H, W).
            indices (Tensor): Indices of selected codebook entries, shape (N,).
        """
        B, D, H, W = z.shape  # (B, D, H, W)

        # Move channels to the end and flatten spatial dimensions:
        # (B, D, H, W) → (B, H, W, D) → (N, D) where N = B * H * W
        z_encoder = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)

        # Full distance matrix between inputs and codebook entries: (N, K)
        distance_matrix = self.codebook.compute_distances(z_encoder)  # (N, K)

        # Find index of the nearest codebook vector for each input vector (non-differentiable)
        encoding_indices = distance_matrix.argmin(dim=1)  # (N,)

        # Retrieve quantized vectors (the chosen codebook entries)
        z_q = self.codebook.lookup(encoding_indices)  # (N, D)

        # Restore spatial layout
        # (N, D) -> (B, H, W, D) -> (B, D, H, W)
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        return z_q, encoding_indices

    def forward(self, x):
        """
        Forward pass through the vector quantizer.

        Args:
            x (Tensor): Input feature map of shape (B, D, H, W).

        Returns:
            z_q (Tensor): Quantized feature map of shape (B, D, H, W).
        """
        z_e = x  # encoder output
        z_q, encoding_indices = self._quantize(x)  # quantizer output

        # codebook loss (updates codebook only)
        # intuition: move codebook entries closer to encoder outputs
        loss_codebook = F.mse_loss(z_q, z_e.detach())

        # commitment loss (updates encoder only)
        # intuition: force encoder to stay near its chosen codebook vector
        loss_commitment = self.beta * F.mse_loss(z_e, z_q.detach())

        # Total VQ loss
        vq_loss = loss_codebook + loss_commitment

        # Straight-through estimator trick for gradient backpropagation
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, vq_loss, encoding_indices
