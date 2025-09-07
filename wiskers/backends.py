"""
This module provides backend-specific inference classes that abstract the loading and sampling behavior of
models saved as:

- PyTorch Lightning Checkpoints (`.ckpt`)
- ONNX Runtime models (`.onnx`)
- SafeTensors serialized models (`.safetensors`)
"""


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


class AECheckpointInference:
    ext = "ckpt"

    def __call__(self, filepath: str, num_samples: int):
        from wiskers.autoencoder.ae_module import AEModule

        self.model = AEModule.load_from_checkpoint(filepath)
        print(f"Load model: {filepath} with hyperparameters:")
        return self.model.generate_samples(num_samples)
