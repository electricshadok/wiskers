import torch.nn as nn

from wiskers.models.world_models.controller import Controller
from wiskers.models.world_models.mdn_rnn import MDNRNN
from wiskers.world_models import WorldModels


def test_world_models_init():
    encoder = nn.Linear(64, 32)
    decoder = nn.Linear(32, 64)
    mdn_rnn = MDNRNN(latent_size=32, action_size=3, hidden_size=256, num_gaussians=5)
    controller = Controller(latent_size=32, hidden_size=256, action_size=3)

    module = WorldModels(
        encoder=encoder,
        decoder=decoder,
        mdn_rnn=mdn_rnn,
        controller=controller,
        learning_rate=1e-3,
    )

    # Encoder and decoder should have requires_grad=False
    for param in module.encoder.parameters():
        assert not param.requires_grad
    for param in module.decoder.parameters():
        assert not param.requires_grad
    for param in module.controller.parameters():
        assert not param.requires_grad

    # MDN-RNN should be trainable
    for param in module.mdn_rnn.parameters():
        assert param.requires_grad

    opt = module.configure_optimizers()
    assert opt is not None

