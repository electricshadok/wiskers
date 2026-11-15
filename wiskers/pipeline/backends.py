"""
This module provides backend-specific inference classes that abstract the loading and sampling behavior of
models saved as:

- PyTorch Lightning Checkpoints (`.ckpt`)
- ONNX Runtime models (`.onnx`)
- SafeTensors serialized models (`.safetensors`)
"""

from typing import Type

import lightning as L


class ONNXInference:
    ext = "onnx"

    def __call__(self, filepath: str, num_samples: int):
        raise NotImplementedError("ONNXInference._call() not implemented")


class SafeTensorInference:
    ext = "safetensors"

    def __call__(self, filepath: str, num_samples: int):
        raise NotImplementedError("SafeTensorInference._call() not implemented")


class CheckpointInference:
    ext = "ckpt"

    def __init__(self, model_class: Type[L.LightningModule]):
        self.model_class = model_class

    def __call__(self, filepath: str, num_samples: int):
        model = self.model_class.load_from_checkpoint(filepath)
        print(f"Loaded model from {filepath} with hyperparameters: {model.hparams}")
        return model.generate_samples(num_samples)
