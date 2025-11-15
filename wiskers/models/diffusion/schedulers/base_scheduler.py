"""
Beta values represent the variance of Gaussian noise added at each step of the forward diffusion process.
These values are small and dictate how much noise is added to the data at each step.
The schedule of β values (how they change over the steps) is critical.
Different beta schedules (like linear, quadratic, etc.) have varying impacts on the diffusion process.

Alpha values are derived from beta values. Specifically, alphe = 1 - βt for each timestep t.
Alphas represent the proportion of the original signal that's retained at each step.
A smaller beta means a larger alpha, indicating more of the original data is preserved in that step.
The cumulative product of alphas is used to calculate the variance of the original data remaining at each timestep.
"""
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from wiskers.models.diffusion.schedulers.beta_schedule import (
    cosine_beta_schedule,
    linear_beta_schedule,
    quadractic_beta_schedule,
    sigmoid_beta_schedule,
)


def extract_into_tensor(x: FloatTensor, t: LongTensor):
    """
    Extracts specific coefficients from a tensor 'x' at specified timesteps t and reshapes the result
    The output tensor will have dimensions [batch_size, 1, 1, 1] for broadcasting purposes.
    Args:
        x (FloatTensor) : (L)
        t (LongTensor) : (B)
    Output:
        (B, 1, 1, 1)
    """
    B = len(t)
    out = x.gather(-1, t)  # (B)
    return out.view(B, 1, 1, 1)


class BaseScheduler(nn.Module):
    """
    Base class for noise scheduler
    Args:
        num_steps (int) : The number of diffusion steps to train the model.
        beta_start (float) : The starting beta value.
        beta_end (float) : The final beta value.
        beta_schedule (str) : defaults to "linear".
    """

    BETA_SCHEDULES = {
        "linear": linear_beta_schedule,
        "quadratic": quadractic_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        beta_schedule: str = "linear",
    ):
        super().__init__()
        if beta_schedule not in self.BETA_SCHEDULES:
            raise ValueError(f"{beta_schedule} is not defined.")

        beta_func = self.BETA_SCHEDULES[beta_schedule]

        self.register_buffer("betas", beta_func(beta_start, beta_end, num_steps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / self.alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # Calculate variance (\sigma_{t}^2)
        # see 3.2 Reverse process from "Denoising diffusion probabilistic models.", 2020
        # two possible variances at time t
        # variance version 1 : self.betas
        # variance version 2 : self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        alphas_cumprod_prev = torch.empty_like(self.alphas_cumprod)
        alphas_cumprod_prev[0] = 1.0
        alphas_cumprod_prev[1:] = self.alphas_cumprod[:-1]
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_start: FloatTensor, t: LongTensor, noise: FloatTensor) -> FloatTensor:
        raise NotImplementedError("q_sample not implemented")

    def p_sample(self, model: nn.Module, x: FloatTensor, t: LongTensor, t_index: int) -> FloatTensor:
        raise NotImplementedError("p_sample not implemented")

    def num_steps(self) -> int:
        return len(self.betas)
