import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Precomputed as a Python float to avoid creating a CPU tensor on every loss call.
LOG_2PI: float = math.log(2.0 * math.pi)


class MDNHead(nn.Module):
    """
    Mixture Density Network (MDN) head.

    Maps an LSTM hidden state to the parameters of a mixture of Gaussians
    over the next latent vector z. This is the 'density' part of the MDN-RNN.

    Args:
        hidden_size (int): Size of the LSTM hidden state.
        latent_size (int): Dimensionality of the latent space z.
        num_gaussians (int): Number of Gaussian components K.

    Shapes:
        in:  [N, T, hidden_size]
        out: pi    [N, T, K]
             mu    [N, T, K, latent_size]
             sigma  [N, T, K, latent_size]
    """

    def __init__(self, hidden_size: int, latent_size: int, num_gaussians: int):
        super().__init__()
        self.latent_size = latent_size
        self.num_gaussians = num_gaussians

        # Single linear that produces all MDN parameters at once.
        # Output size: K (pi) + K * latent_size (mu) + K * latent_size (log_sigma)
        self._fc = nn.Linear(
            hidden_size, num_gaussians + 2 * num_gaussians * latent_size
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h (torch.Tensor): LSTM output of shape [N, T, hidden_size].

        Returns:
            pi    (torch.Tensor): Mixture weights   [N, T, K]
            mu    (torch.Tensor): Gaussian means    [N, T, K, latent_size]
            sigma  (torch.Tensor): Gaussian std devs [N, T, K, latent_size]
        """
        N, T, _ = h.shape
        K = self.num_gaussians
        Z = self.latent_size

        out = self._fc(h)  # [N, T, K + 2*K*Z]

        # Split into (pi_raw, mu, log_sigma) chunks
        pi_raw = out[..., :K]                    # [N, T, K]
        mu_raw = out[..., K : K + K * Z]         # [N, T, K*Z]
        log_sigma_raw = out[..., K + K * Z :]    # [N, T, K*Z]

        pi = F.softmax(pi_raw, dim=-1)
        mu = mu_raw.view(N, T, K, Z)
        # exp(log_sigma_raw) is always > 0, but can get arbitrarily close to 0
        # (e.g. exp(-80) ≈ 1e-35 in float32). Without clamping:
        #   - (z - mu) / sigma explodes to ±Inf → NaN in the loss
        #   - sigma.log() returns -Inf → NaN in the loss
        # Clamping at 1e-6 is a safe lower bound well above float32 underflow.
        sigma = log_sigma_raw.view(N, T, K, Z).exp().clamp(min=1e-6)

        return pi, mu, sigma


class MDNRNN(nn.Module):
    """
    MDN-RNN: the Memory model (M) from the World Models paper (Ha & Schmidhuber, 2018).

    Combines a PyTorch LSTM with a Mixture Density Network (MDN) head. At every
    timestep it consumes the current latent code z_t and action a_t, updates its
    hidden state h_t, and outputs:

    - A mixture-of-Gaussians distribution over the next latent z_{t+1}.
    - A done probability d_{t+1} ∈ [0, 1] predicting episode termination.

    The hidden state h_t is also used as the memory input to the Controller (C).

    Args:
        latent_size (int): Dimensionality of the VAE latent vector z.
        action_size (int): Dimensionality of the action vector a.
        hidden_size (int): Number of units in the LSTM hidden state.
        num_gaussians (int): Number of Gaussian mixture components K. Defaults to 5.
        num_layers (int): Number of stacked LSTM layers. Defaults to 1.

    Shapes:
        in:  z     [N, T, latent_size]
             a     [N, T, action_size]
        out: pi    [N, T, K]
             mu    [N, T, K, latent_size]
             sigma  [N, T, K, latent_size]
             done  [N, T, 1]
             h_n   [num_layers, N, hidden_size]
             c_n   [num_layers, N, hidden_size]
    """

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        hidden_size: int,
        num_gaussians: int = 5,
        num_layers: int = 1,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        self.num_layers = num_layers

        # RNN backbone — reuses PyTorch's optimised LSTM directly
        self._rnn = nn.LSTM(
            input_size=latent_size + action_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # MDN head: hidden state → mixture-of-Gaussians parameters over z_{t+1}
        self._mdn_head = MDNHead(hidden_size, latent_size, num_gaussians)

        # Done head: hidden state → scalar Bernoulli probability of episode end
        self._done_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass over a sequence.

        Args:
            z (torch.Tensor): Latent codes of shape [N, T, latent_size].
            a (torch.Tensor): Actions of shape [N, T, action_size].
            hidden (tuple, optional): Initial (h_0, c_0) for the LSTM.
                                      Defaults to zeros if not provided.

        Returns:
            pi    (torch.Tensor): Mixture weights         [N, T, K]
            mu    (torch.Tensor): Gaussian means          [N, T, K, latent_size]
            sigma  (torch.Tensor): Gaussian std devs       [N, T, K, latent_size]
            done  (torch.Tensor): Episode-end probability  [N, T, 1]
            hidden (tuple):       Updated (h_n, c_n) LSTM states
        """
        # Concatenate latent and action along the feature axis
        rnn_input = torch.cat([z, a], dim=-1)  # [N, T, latent_size + action_size]

        # Run through LSTM — hidden defaults to zeros when None
        rnn_out, hidden = self._rnn(rnn_input, hidden)  # rnn_out: [N, T, hidden_size]

        # MDN: mixture distribution over z_{t+1}
        pi, mu, sigma = self._mdn_head(rnn_out)

        # Done: Bernoulli probability of episode termination at each step
        done = torch.sigmoid(self._done_head(rnn_out))  # [N, T, 1]

        return pi, mu, sigma, done, hidden

    def get_initial_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns zero-initialised (h_0, c_0) LSTM state.

        Args:
            batch_size (int): Number of sequences in the batch.
            device (torch.device): Target device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_0, c_0), each [num_layers, N, hidden_size].
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0

    @staticmethod
    def sample(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample a next latent z from the mixture distribution.

        Uses the reparameterization trick. Temperature < 1 sharpens the
        distribution (more deterministic); temperature > 1 increases randomness.

        Args:
            pi          (torch.Tensor): Mixture weights [N, T, K]
            mu          (torch.Tensor): Means           [N, T, K, Z]
            sigma        (torch.Tensor): Std devs        [N, T, K, Z]
            temperature  (float):        Scales sigma before sampling. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled next latent z of shape [N, T, Z].
        """
        N, T, K, Z = mu.shape

        # Step 1 — pick one Gaussian component per sample via mixture weights
        pi_flat = pi.view(N * T, K)
        k_idx = torch.multinomial(pi_flat, num_samples=1).squeeze(-1)  # [N*T]

        # Step 2 — gather mu and sigma for the chosen component
        k_idx_exp = k_idx.view(N * T, 1, 1).expand(N * T, 1, Z)
        mu_k = mu.view(N * T, K, Z).gather(1, k_idx_exp).squeeze(1)       # [N*T, Z]
        sigma_k = sigma.view(N * T, K, Z).gather(1, k_idx_exp).squeeze(1)  # [N*T, Z]

        # Step 3 — reparameterization sample with temperature scaling
        z_next = mu_k + temperature * sigma_k * torch.randn_like(mu_k)

        return z_next.view(N, T, Z)

    @staticmethod
    def mdn_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss for the MDN output against a target latent z.

        Uses the log-sum-exp trick for numerical stability. Avoids the exp→log
        round-trip by using sigma.log() directly (safe: sigma is clamped ≥ 1e-6).
        Uses a precomputed Python float for log(2π) to avoid CPU tensor creation.

        Args:
            pi       (torch.Tensor): Mixture weights  [N, T, K]
            mu       (torch.Tensor): Means            [N, T, K, latent_size]
            sigma     (torch.Tensor): Std devs         [N, T, K, latent_size]
            z_target  (torch.Tensor): Ground-truth z   [N, T, latent_size]

        Returns:
            torch.Tensor: Scalar mean NLL loss.
        """
        # Expand z_target to match mixture dimension: [N, T, 1, latent_size]
        z_target = z_target.unsqueeze(2)

        # Log-pdf of each Gaussian component (diagonal covariance Gaussian):
        #   log p(z | μ_k, σ_k) = -0.5 * ||( z - μ_k ) / σ_k||² - log(σ_k) - 0.5*log(2π)
        #
        # Numerical stability notes:
        #   • sigma.log() avoids an exp→log round-trip. The naive approach stores
        #     sigma = exp(log_sigma_raw) and then recomputes log(sigma) here, which
        #     loses precision for extreme log_sigma_raw values. Using sigma.log()
        #     directly is exact (up to the clamp) because sigma ≥ 1e-6.
        #   • LOG_2PI is a Python float, so it broadcasts to the correct device/dtype
        #     automatically without allocating any tensor.
        log_prob = (
            -0.5 * ((z_target - mu) / sigma) ** 2
            - sigma.log()
            - 0.5 * LOG_2PI
        )
        # Sum log-probs over the latent dimension (independent dims) → [N, T, K]
        log_prob = log_prob.sum(dim=-1)

        # log(pi + eps) would shift all weights up by eps, biasing the distribution.
        # clamp(min=1e-8) only affects near-zero weights, leaving valid weights exact.
        # pi comes from softmax so it's bounded in (0, 1) — the clamp only guards
        # against the extreme edge case where a component collapses to ~0.
        log_pi = torch.log(pi.clamp(min=1e-8))

        # logsumexp computes log Σ_k π_k * p(z | μ_k, σ_k) in a numerically stable way.
        # The naive approach exp(log_pi + log_prob).sum().log() overflows/underflows
        # when log-probabilities are very large or very negative. logsumexp subtracts
        # the max before exponentiating, keeping values in a safe range.
        log_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)  # [N, T]

        return -log_mixture.mean()

    @staticmethod
    def done_loss(
        done_pred: torch.Tensor,
        done_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss for the done/restart prediction.

        Args:
            done_pred   (torch.Tensor): Predicted episode-end probability [N, T, 1] or [N, T].
            done_target  (torch.Tensor): Binary ground-truth labels         [N, T].

        Returns:
            torch.Tensor: Scalar BCE loss.
        """
        return F.binary_cross_entropy(done_pred.squeeze(-1), done_target.float())
