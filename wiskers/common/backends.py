"""
This module provides backend-specific inference classes that abstract the loading and sampling behavior of
models saved as:

- PyTorch Lightning Checkpoints (`.ckpt`)
- ONNX Runtime models (`.onnx`)
- SafeTensors serialized models (`.safetensors`)
"""

from typing import Type

from wiskers.common.base_module import BaseLightningModule


class ONNXInference:
    ext = "onnx"

    def __call__(self, filepath: str, num_samples: int):
        # TODO - add implementation for ONNXInference
        # import onnxruntime
        # model = onnxruntime.InferenceSession(filepath)
        # inputs = {self.model.get_inputs()[0].name: x}
        # outputs = self.model.run(None, inputs)
        # return outputs[0]
        raise NotImplementedError("ONNXInference._call() not implemented")


class SafeTensorInference:
    ext = "safetensors"

    def __call__(self, filepath: str, num_samples: int):
        # from safetensors.torch import load_model
        # TODO - add implementation for SafeTensorInference
        raise NotImplementedError("SafeTensorInference._call() not implemented")


class CheckpointInference:
    ext = "ckpt"

    def __init__(self, model_class: Type[BaseLightningModule]):
        self.model_class = model_class

    def __call__(self, filepath: str, num_samples: int):
        model = self.model_class.load_from_checkpoint(filepath)
        print(f"Loaded model from {filepath} with hyperparameters: {model.hparams}")
        return model.generate_samples(num_samples)
