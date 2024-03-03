import argparse
import glob
import os

import lightning as L
import onnxruntime
import torchvision

from wiskers.common.commands.utils import load_config
from wiskers.diffusion.diffuser_module import DiffuserModule


class ONNXInference:
    format = "onnx"

    def __init__(self, filepath: str):
        self.model = onnxruntime.InferenceSession(filepath)

    def __call__(self, num_samples: int, num_inference_steps: int):
        # TODO - add implementation for ONNXInference
        # inputs = {self.model.get_inputs()[0].name: x}
        # outputs = self.model.run(None, inputs)
        # return outputs[0]
        raise NotImplementedError("ONNXInference._call() not implemented")


class SafeTensorInference:
    format = "safetensors"

    def __init__(self, filepath: str):
        # from safetensors.torch import load_model
        self.model = None

    def __call__(self, num_samples: int, num_inference_steps: int):
        # TODO - add implementation for SafeTensorInference
        raise NotImplementedError("SafeTensorInference._call() not implemented")


class CheckpointInference:
    format = "ckpt"

    def __init__(self, filepath: str):
        self.model = DiffuserModule.load_from_checkpoint(filepath)
        print(f"Load model: {filepath} with hyperparameters:")
        print(self.model.hparams)

    def __call__(self, num_samples: int, num_inference_steps: int):
        return self.model.generate_samples(num_samples, num_inference_steps)


class GenerateCLI:
    """
    Command-line interface for generating image using a trained PyTorch Lightning model checkpoint.

    Args:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the GenerateCLI.

        Args:
            config_path (str): Path to the configuration file.
        """
        # Load the configuration
        self.config = load_config(config_path, os.path.dirname(__file__))

        # Initialize random number generators
        L.seed_everything(seed=self.config.seed, workers=True)

        # Instance the correct object for inference
        self.model_dir = os.path.join(self.config.best_models_dir, self.config.run_name)
        self.output_dir = self.model_dir  # for now save images in model directory
        model_filepaths = glob.glob(os.path.join(self.model_dir, "*." + self.config.model_format))
        if len(model_filepaths) == 0:
            raise FileNotFoundError(f"No '.{self.config.model_format}' files found in the directory.")

        model_filepath = model_filepaths[0]
        inference_types = [ONNXInference, CheckpointInference, SafeTensorInference]
        inference_formats = [cls.format for cls in inference_types]
        inference_idx = inference_formats.index(self.config.model_format)
        self.inference = inference_types[inference_idx](model_filepath)

    def run(self):
        """
        Runs the image generation process with the user settings.
        """
        samples = self.inference(self.config.num_images, self.config.num_inference_steps)
        output_path = os.path.join(self.output_dir, "output.png")
        torchvision.utils.save_image(samples, output_path)
        # TODO - add FID to evaluate the generated images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training script with a given configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    cmd = GenerateCLI(args.config)
    cmd.run()
