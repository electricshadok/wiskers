import lightning as L


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

    Key ideas vs WorldModels:
        - No separate VAE pre-training; the RSSM is trained end-to-end with
          image decoder + reward decoder as auxiliary losses.
        - Planning is done with CEM (Cross-Entropy Method) in latent space
          at inference time — no explicit policy network.
        - Introduces the deterministic path (GRU) alongside the stochastic
          path for better long-horizon predictions.

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

        # ------------------------------------------------------------------ #
        #  TODO: instantiate RSSM components                                  #
        #    self.rssm      = RSSM(...)       # deterministic + stochastic    #
        #    self.encoder   = ConvEncoder(...)                                 #
        #    self.decoder   = ConvDecoder(...)                                 #
        #    self.reward_model = RewardModel(...)                              #
        # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  TODO: training_step                                                #
    #    1. Encode observations into embeddings with self.encoder.        #
    #    2. Unroll RSSM over the sequence to get (h_t, s_t).             #
    #    3. Decode -> reconstruction loss.                                #
    #    4. Predict reward -> reward loss.                                #
    #    5. KL divergence between posterior and prior.                    #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  TODO: planning (CEM in latent space, inference only)               #
    #    1. Encode current observation -> e_t.                            #
    #    2. Infer current state (h_t, s_t) from RSSM.                    #
    #    3. Run CEM: sample action sequences, simulate in latent space,   #
    #       score by predicted reward sum, return best action.            #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        return __import__("torch").optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
