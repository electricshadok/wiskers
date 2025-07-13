from wiskers.common.datasets.cifar10 import CIFAR10
from wiskers.common.datasets.cifar10_subset import CIFAR10Subset
from wiskers.diffusion.diffuser_module import DiffuserModule
from wiskers.gan.gan_module import GANModule
from wiskers.vae.vae_module import VAEModule


def get_model(model_type: str, **kwargs):
    model = None
    if model_type == "vae":
        model = VAEModule(**kwargs)
    elif model_type == "gan":
        model = GANModule(**kwargs)
    elif model_type == "diffusion":
        model = DiffuserModule(**kwargs)

    return model


def get_data_module(module_type: str, **kwargs):
    data_module = None
    if module_type == "cifar10":
        data_module = CIFAR10(**kwargs)
    elif module_type.startswith("cifar10:"):
        category_name = module_type.split(":")[1]
        data_module = CIFAR10Subset(**kwargs, category_name=category_name)
    else:
        raise ValueError(f"Unsupported data_module_type: {module_type}")
    return data_module
