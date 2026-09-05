from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# RSSMState
# ──────────────────────────────────────────────────────────────────────────────


class RSSMState:
    """
    Container for the RSSM latent state at a single timestep.

    The RSSM (Recurrent State Space Model) decomposes the world-model state
    into two complementary parts:

        h  — deterministic recurrent state produced by a GRU.
             Provides a smooth, lossless memory path across timesteps.
             Shape: [N, hidden_size]

        s  — stochastic state sampled from a Gaussian.
             Captures the irreducible uncertainty in the world.
             Shape: [N, stoch_size]

    Both parts are always kept together and concatenated when feeding the
    decoder, reward predictor, or any downstream model:

        cat(h, s)  →  [N, hidden_size + stoch_size]

    Args:
        h (torch.Tensor): Deterministic state [N, hidden_size].
        s (torch.Tensor): Stochastic state    [N, stoch_size].
    """

    def __init__(self, h: torch.Tensor, s: torch.Tensor) -> None:
        self.h = h
        self.s = s

    @property
    def combined(self) -> torch.Tensor:
        """Concatenation of h and s — used as input to decoders / reward models."""
        return torch.cat([self.h, self.s], dim=-1)  # [N, hidden_size + stoch_size]

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        hidden_size: int,
        stoch_size: int,
        device: torch.device,
    ) -> "RSSMState":
        """Create a zero-initialised state (used at the start of an episode)."""
        h = torch.zeros(batch_size, hidden_size, device=device)
        s = torch.zeros(batch_size, stoch_size, device=device)
        return cls(h, s)


# ──────────────────────────────────────────────────────────────────────────────
# RSSM
# ──────────────────────────────────────────────────────────────────────────────


class RSSM(nn.Module):
    """
    Recurrent State Space Model (RSSM) from PlaNet (Hafner et al., 2019).
    "Learning Latent Dynamics for Planning from Pixels"
    https://arxiv.org/abs/1811.04551

    The RSSM separates the latent state into:
        h_t  — deterministic path  (GRU hidden state)
        s_t  — stochastic path     (Gaussian sample)

    This split lets the model use h_t for smooth long-horizon rollouts while
    s_t captures uncertainty.  Both paths are always used together.

    ── Recurrence (one step) ──────────────────────────────────────────────────

        Input concatenation:
            x_t = cat(s_{t-1}, a_{t-1})          [N, stoch_size + action_size]

        Deterministic transition (GRU):
            h_t = GRU(x_t, h_{t-1})              [N, hidden_size]

        Prior  (used at planning / imagination time — no observation):
            mu_p, logvar_p = prior_net(h_t)
            s_t ~ N(mu_p, sigma_p)

        Posterior (used at training time — observation is available):
            mu_q, logvar_q = posterior_net(cat(h_t, e_t))
            s_t ~ N(mu_q, sigma_q)   where e_t = encoder(o_t)

    ── Training objective (ELBO) ──────────────────────────────────────────────

        L = E[ log p(o_t | h_t, s_t) ]           reconstruction
          + E[ log p(r_t | h_t, s_t) ]           reward prediction
          - KL[ q(s_t | h_t, e_t) || p(s_t | h_t) ]   latent consistency

    Args:
        stoch_size  (int): Dimensionality of the stochastic state s. Default: 32.
        hidden_size (int): Dimensionality of the GRU hidden state h. Default: 256.
        action_size (int): Dimensionality of the action vector a.
        embed_size  (int): Dimensionality of the observation embedding e_t
                           produced by an external CNN encoder. Default: 1024.
        activation  (nn.Module): Activation for prior/posterior MLPs. Default: ELU.

    Shapes (all tensors are single-timestep, batch-first):
        obs_embed   [N, embed_size]
        action      [N, action_size]
        state.h     [N, hidden_size]
        state.s     [N, stoch_size]
        state.combined  [N, hidden_size + stoch_size]
    """

    def __init__(
        self,
        stoch_size: int = 32,
        hidden_size: int = 256,
        action_size: int = 1,
        embed_size: int = 1024,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super().__init__()
        self.stoch_size = stoch_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.embed_size = embed_size

        # ── Deterministic transition ─────────────────────────────────────────
        # GRU input: cat(s_{t-1}, a_{t-1})
        self._gru = nn.GRUCell(
            input_size=stoch_size + action_size,
            hidden_size=hidden_size,
        )

        # ── Prior network: h_t → (mu_p, logvar_p) ───────────────────────────
        # Used during imagination/planning — no observation available.
        self._prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 2 * stoch_size),  # mu and logvar
        )

        # ── Posterior network: cat(h_t, e_t) → (mu_q, logvar_q) ─────────────
        # Used during training — refines the prior with the observed embedding.
        self._posterior_net = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 2 * stoch_size),  # mu and logvar
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: s = mu + eps * std,  eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def _split_mu_logvar(
        raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a [..., 2*stoch_size] tensor into (mu, logvar) halves."""
        mu, logvar = raw.chunk(2, dim=-1)
        return mu, logvar

    # ── Core step ────────────────────────────────────────────────────────────

    def _step_h(self, prev_state: RSSMState, action: torch.Tensor) -> torch.Tensor:
        """
        Advance the deterministic state by one step.

        Args:
            prev_state (RSSMState): State at t-1.
            action     (torch.Tensor): Action a_{t-1} [N, action_size].

        Returns:
            torch.Tensor: New deterministic state h_t [N, hidden_size].
        """
        gru_input = torch.cat([prev_state.s, action], dim=-1)  # [N, stoch + action]
        return self._gru(gru_input, prev_state.h)               # [N, hidden_size]

    def prior_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
    ) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]:
        """
        One RSSM step using the **prior** (no observation).

        Used during imagination / planning rollouts where the agent imagines
        future states without seeing actual observations.

        Args:
            prev_state (RSSMState): State at t-1.
            action     (torch.Tensor): Action a_{t-1} [N, action_size].

        Returns:
            state  (RSSMState):    New state at t.
            mu_p   (torch.Tensor): Prior mean        [N, stoch_size].
            logvar_p (torch.Tensor): Prior log-variance [N, stoch_size].
        """
        h = self._step_h(prev_state, action)                    # [N, hidden_size]
        raw = self._prior_net(h)                                 # [N, 2*stoch_size]
        mu_p, logvar_p = self._split_mu_logvar(raw)
        s = self._reparameterize(mu_p, logvar_p)
        return RSSMState(h, s), mu_p, logvar_p

    def posterior_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[RSSMState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One RSSM step using the **posterior** (observation is available).

        Used during training. The posterior refines the prior estimate using
        the actual observation embedding e_t = encoder(o_t).

        Args:
            prev_state (RSSMState):     State at t-1.
            action     (torch.Tensor):  Action a_{t-1}         [N, action_size].
            obs_embed  (torch.Tensor):  Observation embedding  [N, embed_size].

        Returns:
            state    (RSSMState):    Posterior state at t.
            mu_p     (torch.Tensor): Prior mean        [N, stoch_size].
            logvar_p (torch.Tensor): Prior log-variance [N, stoch_size].
            mu_q     (torch.Tensor): Posterior mean    [N, stoch_size].
            logvar_q (torch.Tensor): Posterior log-variance [N, stoch_size].
        """
        h = self._step_h(prev_state, action)                      # [N, hidden_size]

        # Prior — needed for KL loss even during training
        raw_p = self._prior_net(h)                                # [N, 2*stoch_size]
        mu_p, logvar_p = self._split_mu_logvar(raw_p)

        # Posterior — refined with observation
        raw_q = self._posterior_net(torch.cat([h, obs_embed], dim=-1))
        mu_q, logvar_q = self._split_mu_logvar(raw_q)
        s = self._reparameterize(mu_q, logvar_q)

        return RSSMState(h, s), mu_p, logvar_p, mu_q, logvar_q

    # ── Sequence helpers ─────────────────────────────────────────────────────

    def observe_rollout(
        self,
        obs_embeds: torch.Tensor,
        actions: torch.Tensor,
        init_state: Optional[RSSMState] = None,
    ) -> Tuple[
        "list[RSSMState]",
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Posterior rollout over a full sequence — used during training.

        Unrolls the RSSM over T timesteps using real observation embeddings
        to compute posterior states and collect prior/posterior parameters
        needed for the KL loss.

        Args:
            obs_embeds (torch.Tensor): Observation embeddings [N, T, embed_size].
            actions    (torch.Tensor): Actions                [N, T, action_size].
            init_state (RSSMState, optional): Initial state. Zeros if not provided.

        Returns:
            states   (list[RSSMState]): Posterior states for t = 0..T-1.
            mu_p     (torch.Tensor):    Prior means        [N, T, stoch_size].
            logvar_p (torch.Tensor):    Prior log-vars     [N, T, stoch_size].
            mu_q     (torch.Tensor):    Posterior means    [N, T, stoch_size].
            logvar_q (torch.Tensor):    Posterior log-vars [N, T, stoch_size].
        """
        N, T, _ = obs_embeds.shape
        device = obs_embeds.device

        if init_state is None:
            init_state = RSSMState.zeros(N, self.hidden_size, self.stoch_size, device)

        states = []
        mu_ps, logvar_ps, mu_qs, logvar_qs = [], [], [], []

        state = init_state
        for t in range(T):
            state, mu_p, logvar_p, mu_q, logvar_q = self.posterior_step(
                state, actions[:, t], obs_embeds[:, t]
            )
            states.append(state)
            mu_ps.append(mu_p)
            logvar_ps.append(logvar_p)
            mu_qs.append(mu_q)
            logvar_qs.append(logvar_q)

        return (
            states,
            torch.stack(mu_ps, dim=1),
            torch.stack(logvar_ps, dim=1),
            torch.stack(mu_qs, dim=1),
            torch.stack(logvar_qs, dim=1),
        )

    def imagine_rollout(
        self,
        init_state: RSSMState,
        actions: torch.Tensor,
    ) -> Tuple["list[RSSMState]", torch.Tensor, torch.Tensor]:
        """
        Prior rollout over a future sequence — used for planning / imagination.

        Unrolls using only the prior (no observations). Used by the CEM planner
        at inference time to score imagined action sequences.

        Args:
            init_state (RSSMState):    Starting state.
            actions    (torch.Tensor): Planned actions [N, T, action_size].

        Returns:
            states   (list[RSSMState]): Imagined states for t = 0..T-1.
            mu_p     (torch.Tensor):    Prior means    [N, T, stoch_size].
            logvar_p (torch.Tensor):    Prior log-vars [N, T, stoch_size].
        """
        T = actions.shape[1]

        states = []
        mu_ps, logvar_ps = [], []

        state = init_state
        for t in range(T):
            state, mu_p, logvar_p = self.prior_step(state, actions[:, t])
            states.append(state)
            mu_ps.append(mu_p)
            logvar_ps.append(logvar_p)

        return (
            states,
            torch.stack(mu_ps, dim=1),
            torch.stack(logvar_ps, dim=1),
        )

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def kl_loss(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
        free_nats: float = 3.0,
    ) -> torch.Tensor:
        """
        KL divergence between posterior q and prior p — both diagonal Gaussians.

        KL(q || p) = 0.5 * sum[ log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2) / sigma_p^2 - 1 ]

        The 'free nats' trick (from the paper) clips the KL below a threshold
        so the model doesn't waste capacity compressing very certain states.

        Args:
            mu_q     (torch.Tensor): Posterior mean        [..., stoch_size].
            logvar_q (torch.Tensor): Posterior log-variance [..., stoch_size].
            mu_p     (torch.Tensor): Prior mean            [..., stoch_size].
            logvar_p (torch.Tensor): Prior log-variance    [..., stoch_size].
            free_nats (float): Minimum KL per dimension. Default: 3.0 (from paper).

        Returns:
            torch.Tensor: Scalar mean KL loss.
        """
        # KL between two diagonal Gaussians in closed form
        kl = 0.5 * (
            logvar_p - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1.0
        )
        # Free nats: clamp per-dimension KL below the threshold
        kl = kl.sum(dim=-1)                        # sum over stoch_size → [N, T]
        kl = kl.clamp(min=free_nats)               # free nats trick
        return kl.mean()
