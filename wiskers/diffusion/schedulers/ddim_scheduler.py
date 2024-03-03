"""
Implementation of DDIM
DDIMs can produce high-quality images 10 to 50 times faster in terms compared to DDPMs

Song, Jiaming, Chenlin Meng, and Stefano Ermon.
"Denoising diffusion implicit models."
arXiv preprint arXiv:2010.02502 (2020).
https://arxiv.org/pdf/2010.02502.pdf
"""

from wiskers.diffusion.schedulers.base_scheduler import BaseScheduler


class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.00001,
        beta_end: float = 0.01,
        beta_schedule: str = "linear",
    ):
        super().__init__(num_steps, beta_start, beta_end, beta_schedule)
