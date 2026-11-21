"""
Model-agnostic inference backends for loading serialized artifacts:

- PyTorch Lightning Checkpoints (``.ckpt``)
- SafeTensors serialized models (``.safetensors``)
- ONNX Runtime graphs (``.onnx``)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import lightning as L
import torch.nn as nn
from safetensors.torch import load_model


if TYPE_CHECKING:
    import onnxruntime as ort


class InferenceBackend(ABC):
    """Abstract base class for loading a serialized model artifact for inference."""

    ext: str

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load a model or inference session from ``filepath``."""
        ...


class CheckpointInference(InferenceBackend):
    """Load a trained PyTorch Lightning module checkpoint (``.ckpt``) in eval mode."""

    ext = "ckpt"

    def __init__(self, model_class: type[L.LightningModule], **load_kwargs: Any) -> None:
        self.model_class = model_class
        self.load_kwargs = load_kwargs

    def load(self, filepath: str) -> L.LightningModule:
        model = self.model_class.load_from_checkpoint(filepath, **self.load_kwargs)
        model.eval()
        print(f"Loaded checkpoint from {filepath}")
        return model


class SafeTensorInference(InferenceBackend):
    """Load ``.safetensors`` weights into a PyTorch ``nn.Module`` in eval mode."""

    ext = "safetensors"

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def load(self, filepath: str) -> nn.Module:
        load_model(self.model, filepath)
        self.model.eval()
        return self.model


class ONNXInference(InferenceBackend):
    """Load an ``.onnx`` graph into an ONNX Runtime ``InferenceSession``."""

    ext = "onnx"

    def __init__(self, providers: Sequence[str] | None = None) -> None:
        self.providers = list(providers) if providers is not None else ["CPUExecutionProvider"]

    def load(self, filepath: str) -> "ort.InferenceSession":
        import onnxruntime as ort

        return ort.InferenceSession(filepath, providers=self.providers)
