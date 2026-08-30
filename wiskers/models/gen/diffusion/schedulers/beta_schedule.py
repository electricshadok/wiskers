""" Beta/Variance Schedule
Typically, in diffusion models, the variance changes with each time step, and this can be done in various ways.
I could be like linearly or following a cosine schedule.

Good explainations here
- https://huggingface.co/blog/annotated-diffusion
- Chen, Ting. "On the importance of noise scheduling for diffusion models." arXiv preprint arXiv:2301.10972 (2023).
- https://arxiv.org/pdf/2301.10972.pdf
- https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_02_model.ipynb
"""
import math

import torch


def linear_beta_schedule(beta_start: float, beta_end: float, steps: int) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps)


def quadractic_beta_schedule(beta_start: float, beta_end: float, steps: int) -> torch.Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, steps) ** 2


def sigmoid_beta_schedule(beta_start: float, beta_end: float, steps: int) -> torch.Tensor:
    betas = torch.linspace(-6, 6, steps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return betas


def cosine_beta_schedule(
    beta_start: float,
    beta_end: float,
    steps: int,
    max_beta=0.999,
) -> torch.Tensor:
    betas = []

    def alpha_bar_fn(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    for i in range(steps):
        t1 = i / steps
        t2 = (i + 1) / steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)
