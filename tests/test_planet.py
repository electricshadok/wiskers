import torch.nn as nn

from wiskers.planet import PlaNet


def test_planet_init():
    module = PlaNet(learning_rate=1e-4)
    assert module.learning_rate == 1e-4

    module.dummy = nn.Parameter(nn.Linear(1, 1).weight)
    opt = module.configure_optimizers()
    assert opt is not None

