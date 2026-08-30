import lightning as L
import torch.nn as nn

from wiskers.models.wm2018.controller import Controller
from wiskers.models.wm2018.mdn_rnn import MDNRNN


class WorldModels(L.LightningModule):
    """
    Lightning module for the World Models paper (Ha & Schmidhuber, 2018).

    Architecture:
        V  - Vision model: a VAE that encodes observations into a compact
             latent vector z_t.  The encoder is frozen during Stage 2;
             only the M model is trained here.
        M  - Memory model: an MDN-RNN that learns the transition dynamics
             p(z_{t+1} | z_t, a_t, h_t) in latent space.
        C  - Controller: a single linear layer that maps (z_t, h_t) -> a_t.
             Optimised separately (e.g. CMA-ES), not trained here.

    Training pipeline (3 stages):
        Stage 1 - Train VAE on raw observations (see AEModule / VAEModule).
        Stage 2 - Freeze VAE encoder; train MDN-RNN on encoded sequences.  <- this module
        Stage 3 - Freeze V + M; optimise Controller with CMA-ES in-dream.

    Args:
        encoder (nn.Module): Frozen VAE encoder  z = encoder(x).
        decoder (nn.Module): VAE decoder  x^ = decoder(z).  Kept for
            dream rollouts; not trained here.
        mdn_rnn (MDNRNN): Memory model to train.
        controller (Controller): Controller (C).  Stored for reference /
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

    # ------------------------------------------------------------------ #
    #  TODO: Stage 1 — train VAE separately (AEModule / VAEModule)        #
    #    Load the trained checkpoint and pass encoder/decoder here.       #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  TODO: Stage 2 — training_step / validation_step                    #
    #    1. Encode a sequence of frames with self.encoder (no grad).      #
    #    2. Run self.mdn_rnn on (z_t, a_t) pairs.                         #
    #    3. Compute MDN-NLL loss against z_{t+1} targets.                 #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  TODO: Stage 3 — dream rollout (for Controller optimisation)        #
    #    1. Sample z_0 from the prior.                                    #
    #    2. Loop: a_t = controller(z_t, h_t)                              #
    #             z_{t+1} ~ mdn_rnn(z_t, a_t, h_t)                       #
    #    3. Decode z_t with self.decoder to get dream frames.             #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        return __import__("torch").optim.Adam(
            self.mdn_rnn.parameters(), lr=self.learning_rate
        )
