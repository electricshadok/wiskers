import lightning as L
import torch.nn as nn
from torch.optim import Adam

from wiskers.models.world_models.controller import Controller
from wiskers.models.world_models.mdn_rnn import MDNRNN


class WorldModels(L.LightningModule):
    """
    Lightning module for the World Models paper (Ha & Schmidhuber, 2018).

    Architecture:
        V  - Vision model: a VAE that encodes observations into a compact
             latent vector z_t. The encoder is frozen during Stage 2;
             only the M model is trained here.
        M  - Memory model: an MDN-RNN that learns the transition dynamics
             p(z_{t+1} | z_t, a_t, h_t) in latent space.
        C  - Controller: a single linear layer that maps (z_t, h_t) -> a_t.
             Optimised separately (e.g. CMA-ES), not trained here.

    Training pipeline (3 stages):
        Stage 1 - Train VAE on raw observations (see AEModule / VAEModule).
        Stage 2 - Freeze VAE encoder; train MDN-RNN on encoded sequences.
        Stage 3 - Freeze V + M; optimise Controller with CMA-ES in-dream.

    Args:
        encoder (nn.Module): Frozen VAE encoder  z = encoder(x).
        decoder (nn.Module): VAE decoder  x^ = decoder(z). Kept for
            dream rollouts; not trained here.
        mdn_rnn (MDNRNN): Memory model to train.
        controller (Controller): Controller (C). Stored for reference /
            dream rollouts; weights are not updated by this module.
        learning_rate (float): Learning rate for the MDN-RNN optimiser.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        mdn_rnn: MDNRNN,
        controller: Controller,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder", "mdn_rnn", "controller"])

        # V model — frozen; provides latent codes z_t
        self.encoder = encoder
        self.decoder = decoder
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        for param in self.decoder.parameters():
            param.requires_grad_(False)

        # M model — trained by this module
        self.mdn_rnn = mdn_rnn

        # C model — stored but not trained here
        self.controller = controller
        for param in self.controller.parameters():
            param.requires_grad_(False)

        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return Adam(self.mdn_rnn.parameters(), lr=self.learning_rate)

