from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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

    All heads (ContinuousActor, CategoricalActor, Critic, Reward, Done) share this
    structure — they differ only in what they do with the final linear output.
    """
    layers = []
    current = in_size
    for _ in range(num_layers):
        layers += [nn.Linear(current, hidden_size), activation]
        current = hidden_size
    layers.append(nn.Linear(current, out_size))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Continuous Actor (Squashed Gaussian)
# ──────────────────────────────────────────────────────────────────────────────


class ContinuousActor(nn.Module):
    """
    Continuous Actor (policy) head (e.g. Dreamer, Hafner et al., 2020).
    "Dream to Control: Learning Behaviors by Latent Imagination"
    https://arxiv.org/abs/1912.01603

    Maps a latent/feature vector to a **squashed Gaussian** policy over continuous
    actions. Sampling is differentiable (reparameterization), so gradients flow
    back through the actor during imagination rollouts.

    Architecture:
        feat                                     [N, feat_size]
        MLP(feat)  →  mu_raw, log_std_raw        [N, action_size] each
        std = softplus(log_std_raw).clamp(min, max)
        a   = tanh(mu + std * eps),  eps ~ N(0,I)    ← squashed to (-1, 1)

    Args:
        feat_size   (int): Dimensionality of input feature vector.
        action_size (int): Dimensionality of the continuous action space.
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
            feat (torch.Tensor): Latent feature state [N, feat_size].

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
            feat (torch.Tensor): Latent feature state [N, feat_size].

        Returns:
            action (torch.Tensor): Sampled action [N, action_size] ∈ (-1, 1).
            mu     (torch.Tensor): Squashed mean  [N, action_size].
        """
        mu, std = self(feat)
        eps = torch.randn_like(std)
        action = torch.atanh(mu.clamp(-0.999, 0.999)) + eps * std
        action = torch.tanh(action)
        return action, mu


# ──────────────────────────────────────────────────────────────────────────────
# Categorical Actor (Discrete Actions)
# ──────────────────────────────────────────────────────────────────────────────


class CategoricalActor(nn.Module):
    """
    Discrete Categorical Actor (policy) head for discrete action spaces (e.g. Atari).

    Maps a latent/feature vector to unnormalized action logits over ``num_actions``
    discrete choices and provides sampling with log-probabilities and entropy for
    policy-gradient (REINFORCE / Actor-Critic) updates.

    Args:
        feat_size   (int): Dimensionality of input feature vector.
        num_actions (int): Number of discrete actions.
        hidden_size (int): Hidden layer width. Default: 400.
        num_layers  (int): Number of hidden layers. Default: 4.
        activation  (nn.Module): Activation function. Default: ELU.

    Shapes:
        in:  feat   [N, feat_size]
        out: logits [N, num_actions]
    """

    def __init__(
        self,
        feat_size: int,
        num_actions: int,
        hidden_size: int = 400,
        num_layers: int = 4,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self._mlp = _build_mlp(feat_size, hidden_size, num_layers, num_actions, activation)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.

        Args:
            feat (torch.Tensor): Latent feature state [..., feat_size].

        Returns:
            torch.Tensor: Action logits [..., num_actions].
        """
        return self._mlp(feat)

    def dist(self, feat: torch.Tensor) -> Categorical:
        """Return a Categorical distribution over actions."""
        return Categorical(logits=self(feat))

    def sample(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a discrete action and return its log-probability and distribution entropy.

        Args:
            feat (torch.Tensor): Latent feature state [N, feat_size].

        Returns:
            action   (torch.Tensor): Sampled action indices [N] (long).
            log_prob (torch.Tensor): Log-probability log π(a | s) [N].
            entropy  (torch.Tensor): Policy entropy H(π(· | s)) [N].
        """
        d = self.dist(feat)
        action = d.sample()
        return action, d.log_prob(action), d.entropy()


# ──────────────────────────────────────────────────────────────────────────────
# Critic
# ──────────────────────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """
    Critic (value function) head.

    Estimates the expected discounted return V(s_t) from the current latent/feature
    state.

    Architecture:
        feat       [N, feat_size]
        MLP(feat)  →  V  [N, 1]

    Args:
        feat_size   (int): Dimensionality of input feature vector.
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
            feat (torch.Tensor): Latent feature state [N, feat_size].

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
        dones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute lambda-returns over an imagined trajectory.

        Lambda-returns interpolate between 1-step TD (lam=0) and Monte Carlo
        returns (lam=1):

            γ_t   = γ * (1 - d_t)
            G_t^λ = r_t + γ_t * [ (1 - λ) V(s_{t+1}) + λ G_{t+1}^λ ]

        Args:
            rewards (torch.Tensor): Imagined rewards [T, N, 1] (or [T, N]).
            values  (torch.Tensor): Critic value estimates [T + 1, N, 1] (or [T + 1, N]).
                The last entry values[T] is the bootstrap value for the final step.
            gamma   (float): Discount factor. Default: 0.99.
            lam     (float): Lambda for TD(λ). Default: 0.95.
            dones   (Optional[torch.Tensor]): Termination flags or probabilities
                in [0, 1] of same shape as ``rewards``. When ``dones[t] == 1``,
                future bootstrap values are masked out (``G_t^λ = r_t``).

        Returns:
            torch.Tensor: Lambda-returns of same shape as ``rewards``.
        """
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)

        # Bootstrap from the last value estimate
        next_return = values[T]

        for t in reversed(range(T)):
            if dones is not None:
                gamma_t = gamma * (1.0 - dones[t].to(rewards.dtype))
            else:
                gamma_t = gamma

            td = rewards[t] + gamma_t * values[t + 1]
            next_return = td + gamma_t * lam * (next_return - values[t + 1])
            returns[t] = next_return

        return returns


# ──────────────────────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────────────────────


class Reward(nn.Module):
    """
    Reward prediction head.

    Predicts the scalar reward r_t from a latent/feature state.

    Architecture:
        feat       [N, feat_size]
        MLP(feat)  →  r_hat  [N, 1]

    Args:
        feat_size   (int): Dimensionality of input feature vector.
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
            feat (torch.Tensor): Latent feature state [N, feat_size].

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


# ──────────────────────────────────────────────────────────────────────────────
# Done (Episode Termination Head)
# ──────────────────────────────────────────────────────────────────────────────


class Done(nn.Module):
    """
    Episode termination (done) prediction head.

    Predicts the unnormalized logit for episode termination d_t ∈ {0, 1} from
    a latent/feature state.

    Architecture:
        feat       [N, feat_size]
        MLP(feat)  →  done_logit  [N, 1]

    Args:
        feat_size   (int): Dimensionality of input feature vector.
        hidden_size (int): Hidden layer width. Default: 400.
        num_layers  (int): Number of hidden layers. Default: 4.
        activation  (nn.Module): Activation function. Default: ELU.

    Shapes:
        in:  feat        [N, feat_size]
        out: done_logit  [N, 1]
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
        Predict termination logits from the latent state.

        Args:
            feat (torch.Tensor): Latent feature state [N, feat_size].

        Returns:
            torch.Tensor: Termination logit [N, 1].
        """
        return self._mlp(feat)

    def predict_proba(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Predict termination probability P(done = 1 | feat) ∈ (0, 1).

        Args:
            feat (torch.Tensor): Latent feature state [N, feat_size].

        Returns:
            torch.Tensor: Termination probability [N, 1].
        """
        return torch.sigmoid(self(feat))

    @staticmethod
    def loss(done_logits: torch.Tensor, done_target: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy loss with logits between predicted termination logits
        and ground-truth termination flags.

        Args:
            done_logits (torch.Tensor): Predicted logits [N, 1] or [N, T, 1].
            done_target (torch.Tensor): Ground-truth binary targets (0/1 or bool), same shape.

        Returns:
            torch.Tensor: Scalar mean BCE loss.
        """
        return F.binary_cross_entropy_with_logits(done_logits, done_target.to(done_logits.dtype))

