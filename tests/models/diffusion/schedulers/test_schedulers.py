import pytest

from wiskers.models.diffusion.schedulers.ddim_scheduler import DDIMScheduler
from wiskers.models.diffusion.schedulers.ddpm_scheduler import DDPMScheduler


@pytest.mark.parametrize(
    "num_steps, beta_start, beta_end, beta_schedule",
    [
        (500, 1e-5, 1e-2, "linear"),
        (750, 1e-5, 1e-2, "quadratic"),
        (1000, 1e-5, 1e-2, "cosine"),
        (1000, 1e-5, 1e-2, "sigmoid"),
    ],
)
def test_schedulers(num_steps, beta_start, beta_end, beta_schedule):
    scheduler_types = [DDIMScheduler, DDPMScheduler]

    for scheduler_type in scheduler_types:
        scheduler = scheduler_type(num_steps, beta_start, beta_end, beta_schedule)

        assert scheduler.alphas.shape == (num_steps,)
        assert scheduler.betas.shape == (num_steps,)
        assert scheduler.alphas_cumprod.shape == (num_steps,)
        assert scheduler.sqrt_alphas_cumprod.shape == (num_steps,)
        assert scheduler.sqrt_one_minus_alphas_cumprod.shape == (num_steps,)
