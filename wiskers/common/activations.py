from typing import Type

import torch.nn as nn


class ActivationFct:
    _registry = {
        "relu": nn.ReLU(),  # Rectified Linear Unit
        "leakyrelu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh" : nn.Tanh(),
        "softplus": nn.Softplus(),
        "gelu": nn.GELU(),  # Gaussian Error Linear Unit
        "silu": nn.SiLU(),  # Sigmoid Linear Unit (also known as Swish)
    }

    @staticmethod
    def register(key, value):
        ActivationFct._registry[key] = value

    @staticmethod
    def get(key) -> Type[nn.Module]:
        if key not in ActivationFct._registry:
            raise ValueError(f"Item not found for key: {key}")
        return ActivationFct._registry[key]
