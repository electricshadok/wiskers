from wiskers.common.datasets.cifar10 import CIFAR10
from wiskers.common.datasets.cifar10_subset import CIFAR10Subset
from wiskers.common.datasets.clevrer import CLEVRER
from wiskers.diffusion.diffuser_module import DiffuserModule
from wiskers.gan.gan_module import GANModule
from wiskers.vae.vae_module import VAEModule


def get_model(model_type: str, **kwargs):
    if model_type == "vae":
        return VAEModule(**kwargs)
    elif model_type == "gan":
        return GANModule(**kwargs)
    elif model_type == "diffusion":
        return DiffuserModule(**kwargs)

    raise ValueError(f"Unsupported model_type: {model_type}")


def get_data_module(module_type: str, **kwargs):
    if module_type == "cifar10":
        return CIFAR10(**kwargs)
    elif module_type.startswith("cifar10:"):
        category_name = module_type.split(":")[1]
        return CIFAR10Subset(**kwargs, category_name=category_name)
    elif module_type == "clevrer":
        return CLEVRER(**kwargs)

    raise ValueError(f"Unsupported data_module_type: {module_type}")
