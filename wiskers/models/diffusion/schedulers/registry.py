from wiskers.models.diffusion.schedulers.base_scheduler import BaseScheduler
from wiskers.models.diffusion.schedulers.ddim_scheduler import DDIMScheduler
from wiskers.models.diffusion.schedulers.ddpm_scheduler import DDPMScheduler


class Schedulers:
    _registry = {
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler,
    }

    @staticmethod
    def register(key, value):
        Schedulers._registry[key] = value

    @staticmethod
    def get(key: str) -> BaseScheduler:
        if key not in Schedulers._registry:
            raise ValueError(f"Item not found for key: {key}")
        return Schedulers._registry[key]
