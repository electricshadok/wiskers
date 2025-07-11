from wiskers.common.datasets.cifar10 import CIFAR10
from wiskers.common.datasets.cifar10_subset import CIFAR10Subset


# from wiskers.gan.gan_module import GANModule
# from wiskers.vae.vae_module import VAEModule


# TODO : Implement get_model() : "diffusion"


def get_data_module(module_type: str, config: dict):
    data_module = None
    if module_type == "cifar10":
        data_module = CIFAR10(**config)
    elif module_type.startswith("cifar10:"):
        category_name = module_type.split(":")[1]
        data_module = CIFAR10Subset(**config, category_name=category_name)
    else:
        raise ValueError(f"Unsupported data_module_type: {module_type}")
    return data_module
