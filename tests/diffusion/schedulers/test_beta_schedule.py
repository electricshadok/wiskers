import pytest
import torch

from wiskers.diffusion.schedulers.beta_schedule import linear_beta_schedule


@pytest.mark.parametrize(
    "beta_start, beta_end, steps",
    [
        (1e-5, 1e-2, 500),
        (1e-5, 1e-2, 1000),
    ],
)
def test_linear_beta_scheduler(beta_start, beta_end, steps):
    t = linear_beta_schedule(beta_start, beta_end, steps)

    assert isinstance(t, torch.Tensor)
    assert t.shape == (steps,)
    assert t[0] == beta_start
    assert t[-1] == beta_end
    assert t.dtype == torch.float32
