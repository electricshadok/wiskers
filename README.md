# Wiskers: World Models & Policy Learning Playground

![Wiskers Logo](https://raw.githubusercontent.com/electricshadok/wiskers/refs/heads/main/docs/wiskers_logo.png)

Wiskers is a modular framework to build, train, and evaluate world models and policies in simulation. Starting with core world model and policy training, Wiskers is designed to progressively support diverse architectures for reinforcement learning and generative dynamics—including latent world models, diffusion policies, Vision-Language-Action models (VLAs), behavior cloning, and actor-critic RL—providing an experimental playground to train policies and benchmark agent intelligence from imagination to simulated execution.

> 📘 For the full philosophy and system blueprint, see [PROJECT.md](https://github.com/electricshadok/wiskers/blob/main/PROJECT.md)

## Installation

Create and activate your environment:

```bash
mamba env create -f environment.yml
mamba activate wiskers
```

## Tests

Run unit tests:

```bash
pytest tests
```

### Development Runs

`--fast_dev_run`

Use this for the fastest sanity check that everything compiles and runs without executing the full training pipeline:

```bash
python wiskers/cli/train.py --config <config_yaml> --fast_dev_run
```

`--quick_run`

Use this to simulate a short training run and catch bugs across the full pipeline (checkpointing, logging, etc.):

```bash
python wiskers/cli/train.py --config <config_yaml> --quick_run
```

## Training

Launch training with a chosen experiment configuration file (`<config_yaml>`):

```bash
python wiskers/cli/train.py --config <config_yaml>
```

## Prediction & Evaluation

Run evaluation or inference with a configuration:

```bash
python wiskers/cli/predict.py --config <config_yaml>
```

## Inspect Datasets

Inspect and verify trajectory dataloaders and rollout replay buffers:

```bash
# Atari Breakout trajectory dataloader
python wiskers/cli/inspect_dataset.py --config configs/datasets/breakout.yaml

# CarRacing-v3 trajectory dataloader
python wiskers/cli/inspect_dataset.py --config configs/datasets/carracing.yaml

# Franka Kitchen robotics manipulator trajectory dataloader
python wiskers/cli/inspect_dataset.py --config configs/datasets/franka.yaml
```

## Streamlit WebUI

Streamlit provides an interactive WebUI to explore rollout trajectories, transitions, and models:

```bash
./streamlit.sh
```

<details>
<summary>Screenshot</summary>
<p align="center"><img src="docs/app.png?raw=true"></p>
</details>

## Tensorboard

Track ML experiments and training metrics:

```bash
tensorboard --logdir=experiments/ --port=6006
```

## Docker

### Build Docker container

```bash
./build_docker.sh
```

### Start Docker container

```bash
docker run --rm -it wiskers:1.0
```

## Code Formatter

Check and format code with ruff (configured in `pyproject.toml`):

```bash
ruff check --fix
```

## License

Wiskers is licensed under the MIT License — see the [LICENSE](https://github.com/electricshadok/wiskers/blob/main/LICENSE) file for details.
