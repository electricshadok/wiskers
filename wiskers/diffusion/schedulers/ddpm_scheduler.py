"""
Implementation of DDPM & Improved DDPM

[1] DDPM
Ho, Jonathan, Ajay Jain, and Pieter Abbeel.
"Denoising diffusion probabilistic models.", 2020
Advances in neural information processing systems 33 (2020): 6840-6851.
https://arxiv.org/pdf/2006.11239.pdf

[2] Improved DDPM
Nichol, Alexander Quinn, and Prafulla Dhariwal.
"Improved denoising diffusion probabilistic models."m 2021
In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021.
https://arxiv.org/pdf/2102.09672.pdf
"""
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from wiskers.diffusion.schedulers.base_scheduler import BaseScheduler, extract_into_tensor


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.00001,
        beta_end: float = 0.01,
        beta_schedule: str = "linear",
    ):
        super().__init__(num_steps, beta_start, beta_end, beta_schedule)

    def q_sample(self, x_start: FloatTensor, t: LongTensor, noise: FloatTensor) -> FloatTensor:
        """
        See Algorithm 1 from "Denoising diffusion probabilistic models.", 2020
        """
        a = extract_into_tensor(self.sqrt_alphas_cumprod, t)
        b = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t)
        # a and b are of shape (B, 1, 1, 1)
        return x_start * a + noise * b

    def p_sample(self, model: nn.Module, x: FloatTensor, t: LongTensor, t_index: int) -> FloatTensor:
        """
        See Algorithm 2 from "Denoising diffusion probabilistic models.", 2020
        """
        # Calculate the model at time t (\mu_{t})
        # use Equation 11 from "Denoising diffusion probabilistic models.", 2020
        betas_t = extract_into_tensor(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = extract_into_tensor(self.sqrt_recip_alphas, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        # Get the model variance at time t (\sigma_{t}^2)
        if t_index == 0:
            return model_mean

        posterior_variance_t = extract_into_tensor(self.posterior_variance, t)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
