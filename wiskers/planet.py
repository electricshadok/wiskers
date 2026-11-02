import lightning as L
from torch.optim import Adam


class PlaNet(L.LightningModule):
    """
    Lightning module for PlaNet (Hafner et al., 2019).
    "Learning Latent Dynamics for Planning from Pixels"
    https://arxiv.org/abs/1811.04551

    Architecture:
        RSSM - Recurrent State Space Model: joint recurrent + stochastic
               state that separates deterministic (h_t) and stochastic (s_t)
               components.
        Encoder  - Encodes pixel observations o_t into embedded features e_t.
        Decoder  - Reconstructs observations from the latent state (s_t, h_t).
        Reward   - Predicts reward r_t from the latent state.

    Training objective (single stage, joint):
        L = L_reconstruction + L_reward + L_KL

    Args:
        learning_rate (float): Learning rate for the joint optimiser.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

