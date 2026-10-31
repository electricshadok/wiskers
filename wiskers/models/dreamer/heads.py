from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared MLP builder
# ──────────────────────────────────────────────────────────────────────────────


def _build_mlp(
    in_size: int,
    hidden_size: int,
    num_layers: int,
    out_size: int,
    activation: nn.Module,
) -> nn.Sequential:
    """
    Build a simple MLP: Linear → Activation → ... → Linear (no final activation).

    All three heads (Actor, Critic, Reward) share this structure — they differ
    only in what they do with the final linear output.
    """
    layers = []
    current = in_size
    for _ in range(num_layers):
        layers += [nn.Linear(current, hidden_size), activation]
        current = hidden_size
    layers.append(nn.Linear(current, out_size))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Actor
# ──────────────────────────────────────────────────────────────────────────────


class Actor(nn.Module):
    """
    Actor (policy) head from Dreamer (Hafner et al., 2020).
    "Dream to Control: Learning Behaviors by Latent Imagination"
    https://arxiv.org/abs/1912.01603

    Maps the RSSM latent state to a **squashed Gaussian** policy over continuous
    actions. The key design choice vs a deterministic policy is that sampling is
    differentiable (reparameterization), so gradients flow back through the actor
    during imagination rollouts.

    Architecture:
        feat = cat(h_t, s_t)                     [N, feat_size]
        MLP(feat)  →  mu_raw, log_std_raw        [N, action_size] each
        std = softplus(log_std_raw).clamp(min, max)
        a   = tanh(mu + std * eps),  eps ~ N(0,I)    ← squashed to (-1, 1)

    The tanh squashing keeps actions bounded in (-1, 1)^action_size, which is
    standard for continuous control. The log_std is clamped to prevent the
    policy from collapsing to deterministic or becoming too noisy.

    Args:
        feat_size   (int): Dimensionality of cat(h, s) from the RSSM.
        action_size (int): Dimensionality of the action space.
        hidden_size (int): Hidden layer width. Default: 400.
        num_layers  (int): Number of hidden layers. Default: 4.
        min_std     (float): Minimum policy standard deviation. Default: 1e-4.
        max_std     (float): Maximum policy standard deviation. Default: 1.0.
        activation  (nn.Module): Activation function. Default: ELU.

    Shapes:
        in:  feat  [N, feat_size]
        out: mu    [N, action_size]   (squashed mean, for deterministic eval)
             std   [N, action_size]   (policy spread)
    """

    def __init__(
        self,
        feat_size: int,
        action_size: int,
        hidden_size: int = 400,
        num_layers: int = 4,
        min_std: float = 1e-4,
        max_std: float = 1.0,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        # Outputs mu_raw and log_std_raw concatenated → split in forward
        self._mlp = _build_mlp(feat_size, hidden_size, num_layers, 2 * action_size, activation)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the squashed Gaussian policy parameters.

        Args:
            feat (torch.Tensor): RSSM latent state cat(h, s) [N, feat_size].

        Returns:
            mu  (torch.Tensor): Squashed mean action  [N, action_size]  ∈ (-1, 1).
            std (torch.Tensor): Policy std dev        [N, action_size]  > 0.
        """
        raw = self._mlp(feat)                               # [N, 2 * action_size]
        mu_raw, log_std_raw = raw.chunk(2, dim=-1)          # [N, action_size] each

        # Squash mean to (-1, 1) — deterministic action used at eval time
        mu = torch.tanh(mu_raw)

        # Softplus ensures std > 0; clamp keeps it in a stable range
        std = F.softplus(log_std_raw).clamp(self.min_std, self.max_std)

        return mu, std

    def sample(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the squashed Gaussian policy (reparameterization).

        Gradients flow through the sampled action back into the actor weights,
        enabling backpropagation through imagined rollouts.

        Args:
            feat (torch.Tensor): RSSM latent state [N, feat_size].

        Returns:
            action (torch.Tensor): Sampled action [N, action_size] ∈ (-1, 1).
            mu     (torch.Tensor): Squashed mean  [N, action_size].
        """
        mu, std = self(feat)
        # Reparameterization: tanh(mu_raw + eps * std_raw) ≈ tanh(Normal sample)
        # Here we approximate by sampling in tanh-space around mu
        eps = torch.randn_like(std)
        action = (torch.atanh(mu.clamp(-0.999, 0.999)) + eps * std)
        action = torch.tanh(action)
        return action, mu


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """
    Critic (value function) head from Dreamer (Hafner et al., 2020).

    Estimates the expected discounted return V(s_t) from the current RSSM
    state. Trained entirely inside latent space using lambda-returns computed
    over imagined rollouts — it never sees actual pixel observations.

    Architecture:
        feat = cat(h_t, s_t)      [N, feat_size]
        MLP(feat)  →  V           [N, 1]

    The scalar output is NOT squashed — value estimates can be any real number.

    Args:
        feat_size   (int): Dimensionality of cat(h, s) from the RSSM.
        hidden_size (int): Hidden layer width. Default: 400.
        num_layers  (int): Number of hidden layers. Default: 4.
        activation  (nn.Module): Activation function. Default: ELU.

    Shapes:
        in:  feat  [N, feat_size]
        out: value [N, 1]
    """

    def __init__(
        self,
        feat_size: int,
        hidden_size: int = 400,
        num_layers: int = 4,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super().__init__()
        self._mlp = _build_mlp(feat_size, hidden_size, num_layers, 1, activation)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Estimate the state value.

        Args:
            feat (torch.Tensor): RSSM latent state cat(h, s) [N, feat_size].

        Returns:
            torch.Tensor: Estimated value V(s) [N, 1].
        """
        return self._mlp(feat)  # [N, 1]

    @staticmethod
    def lambda_return(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute lambda-returns over an imagined trajectory.

        Lambda-returns interpolate between 1-step TD (lam=0) and Monte Carlo
        returns (lam=1). The paper uses lam=0.95 as a good bias-variance tradeoff.

        G_t^λ = r_t + γ [ (1-λ) V(s_{t+1}) + λ G_{t+1}^λ ]

        Args:
            rewards (torch.Tensor): Imagined rewards     [T, N, 1].
            values  (torch.Tensor): Critic value estimates [T+1, N, 1].
                The last entry values[T] is the bootstrap value for the final step.
            gamma (float): Discount factor. Default: 0.99.
            lam   (float): Lambda for TD(λ). Default: 0.95.

        Returns:
            torch.Tensor: Lambda-returns [T, N, 1].
        """
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)

        # Bootstrap from the last value estimate
        next_return = values[T]

        for t in reversed(range(T)):
            td = rewards[t] + gamma * values[t + 1]
            next_return = td + gamma * lam * (next_return - values[t + 1])
            returns[t] = next_return

        return returns


# ──────────────────────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────────────────────


class Reward(nn.Module):
    """
    Reward prediction head from Dreamer (Hafner et al., 2020).

    Predicts the scalar reward r_t from the RSSM latent state. Trained jointly
    with the world model (RSSM) as an auxiliary reconstruction target, giving
    the model a signal to capture reward-relevant structure in the latent space.

    Architecture:
        feat = cat(h_t, s_t)      [N, feat_size]
        MLP(feat)  →  r_hat       [N, 1]

    Args:
        feat_size   (int): Dimensionality of cat(h, s) from the RSSM.
        hidden_size (int): Hidden layer width. Default: 400.
        num_layers  (int): Number of hidden layers. Default: 4.
        activation  (nn.Module): Activation function. Default: ELU.

    Shapes:
        in:  feat   [N, feat_size]
        out: r_hat  [N, 1]
    """

    def __init__(
        self,
        feat_size: int,
        hidden_size: int = 400,
        num_layers: int = 4,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super().__init__()
        self._mlp = _build_mlp(feat_size, hidden_size, num_layers, 1, activation)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Predict the reward from the latent state.

        Args:
            feat (torch.Tensor): RSSM latent state cat(h, s) [N, feat_size].

        Returns:
            torch.Tensor: Predicted reward r_hat [N, 1].
        """
        return self._mlp(feat)  # [N, 1]

    @staticmethod
    def loss(r_hat: torch.Tensor, r_target: torch.Tensor) -> torch.Tensor:
        """
        MSE loss between predicted and actual rewards.

        Args:
            r_hat     (torch.Tensor): Predicted reward [N, 1] or [N, T, 1].
            r_target  (torch.Tensor): Ground-truth reward, same shape.

        Returns:
            torch.Tensor: Scalar mean MSE loss.
        """
        return F.mse_loss(r_hat, r_target)
