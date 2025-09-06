import argparse
import glob
import os

import lightning as L
import onnxruntime
import torchvision
from hydra.utils import instantiate

from wiskers.autoencoder.ae_module import AEModule
from wiskers.common.commands.utils import load_config


class ONNXInference:
    format = "onnx"

    def __init__(self, filepath: str):
        self.model = onnxruntime.InferenceSession(filepath)

    def __call__(self, num_samples: int):
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

    def __call__(self, num_samples: int):
        # TODO - add implementation for SafeTensorInference
        raise NotImplementedError("SafeTensorInference._call() not implemented")


class CheckpointInference:
    format = "ckpt"

    def __init__(self, filepath: str):
        self.model = AEModule.load_from_checkpoint(filepath)
        print(f"Load model: {filepath} with hyperparameters:")

    def __call__(self, num_samples: int):
        return self.model.generate_samples(num_samples)


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
        self.config = load_config(config_path)

        # Initialize random number generators
        L.seed_everything(seed=self.config.seed, workers=True)

        model_filepath = instantiate(self.config.best_model_path)

        inference_types = [ONNXInference, CheckpointInference, SafeTensorInference]
        inference_formats = [cls.format for cls in inference_types]
        ext = os.path.splitext(model_filepath)[1][1:]
        inference_idx = inference_formats.index(ext)
        self.inference = inference_types[inference_idx](model_filepath)

    @staticmethod
    def get_output_dir(best_models_dir: str, run_name: str) -> str:
        run_dir = os.path.join(best_models_dir, run_name)
        output_dir = os.path.join(run_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def get_model_path(
        best_models_dir: str, run_name: str | None, model_format: str
    ) -> str:
        # Instance the correct object for inference
        if run_name is None:
            run_name = GenerateCLI.find_latest_run(best_models_dir)
        run_dir = os.path.join(best_models_dir, run_name)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"{checkpoint_dir} doesn't exist")

        model_filepaths = glob.glob(os.path.join(checkpoint_dir, "*." + model_format))
        if len(model_filepaths) == 0:
            raise FileNotFoundError(
                f"No '.{model_format}' files found in {checkpoint_dir}"
            )

        return model_filepaths[-1]  # For now get the latest checkpoint

    @staticmethod
    def find_latest_run(best_models_dir: str):
        """
        Returns the subdirectory in `base_dir` with the highest numeric suffix in the format <prefix>_<number>.
        Example: among ['run_1', 'run_2', 'run_3'], returns 'run_3'.
        """
        max_suffix = -1
        latest_run = None

        for name in os.listdir(best_models_dir):
            full_path = os.path.join(best_models_dir, name)
            if not os.path.isdir(full_path):
                continue

            if "_" in name:
                prefix, suffix = name.rsplit("_", 1)
                if suffix.isdigit():
                    num = int(suffix)
                    if num > max_suffix:
                        max_suffix = num
                        latest_run = name

        return latest_run

    def run(self):
        """
        Runs the image generation process with the user settings.
        """
        output_dir = instantiate(self.config.output_dir)
        samples = self.inference(self.config.num_images)
        output_path = os.path.join(output_dir, "output.png")
        torchvision.utils.save_image(samples, output_path)
        print(f"Export output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training script with a given configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()
    cmd = GenerateCLI(args.config)
    cmd.run()
