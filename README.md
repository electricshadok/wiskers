
![PyTorch Logo](https://raw.githubusercontent.com/electricshadok/wiskers/refs/heads/main/docs/wiskers_logo.png)

--------------------------------------------------------------------------------


**Wiskers: DL and  Generative AI Study**


This repository serves as a learning hub for experimenting with foundational deep learning (DL) architectures and Generative AI. Currently, I am exploring a variety of deep learning techniques (diffusion models, VAEs, GANs) to generate images, videos, audio, and animations into a single, cohesive resource. Expect this repo to continuously evolve.

Fair warning: My curiosity may lead this repo down multiple paths, but fear not—a windy journey is just another step towards clarity. Let’s have fun!

## Installation

Create and activate your environment.
```
mamba env create -f environment.yml
```

```
mamba activate whiskers_dev
```

## Tests

Run unit tests

```
pytest tests
```

```--fast_dev_run```: Run a quick training process.

```
# CLEVRER
python wiskers/train.py --config configs/train_clevrer_stg1.yaml --fast_dev_run

# CIFAR-10
python wiskers/train.py --config configs/models/diffusion/train.yaml --fast_dev_run
python wiskers/train.py --config configs/models/vae/train.yaml --fast_dev_run
python wiskers/train.py --config configs/models/gan/train.yaml --fast_dev_run
```

```--quick_run``` : Run few batches and single epoch.

```
# CLEVRER
python wiskers/train.py --config configs/train_clevrer_stg1.yaml --quick_run

# CIFAR-10
python wiskers/train.py --config configs/models/diffusion/train.yaml --quick_run
python wiskers/train.py --config configs/models/vae/train.yaml --quick_run
python wiskers/train.py --config configs/models/gan/train.yaml --quick_run
```

## Training

Use the following command to begin the training process. The configuration is defined in *train.yaml*, which can be tailored to suit your specific training needs.

```
# CLEVRER
python wiskers/train.py --config configs/train_clevrer_stg1.yaml

# CIFAR-10
python wiskers/train.py --config configs/models/diffusion/train.yaml
python wiskers/train.py --config configs/models/vae/train.yaml
python wiskers/train.py --config configs/models/gan/train.yaml
```

## Inference

Use trained model to generate samples. This command triggers the sample generation proces with the configuration *generate.yaml*.

```
# CIFAR-10
python wiskers/generate.py --config configs/models/diffusion/generate.yaml
python wiskers/generate.py --config configs/models/vae/generate.yaml
python wiskers/generate.py --config configs/models/gan/generate.yaml
```

## Inspect Datasets

Inspect and test the dataloaders.

```
python wiskers/inspect_dataset.py --config configs/datasets/clevrer_image.yaml
python wiskers/inspect_dataset.py --config configs/datasets/clevrer_video.yaml
python wiskers/inspect_dataset.py --config configs/datasets/cifar10.yaml
```

## Streamlit

Streamlit is used to debug and run diffusion code, facilitating a smoother development and testing process.

```
streamlit run streamlit_ui/main.py
```

<details>
<summary>Screenshot</summary>
<p align="center"><img src="docs/app.png?raw=true"></p>
</details>


## Tensorboard

Track ML experiments

```
tensorboard --logdir=experiments/ --port=6006
```

## Docker

### build docker container

Docker ensures a consistent environment Docker. Simply run the command to build the Docker container:

```
./build_docker.sh
```

### start docker container
Once the build is complete, you can start the Docker container with all necessary dependencies installed:

```
docker run --rm -it wiskers:1.0
```

## Code Formatter
Use code formatter with ruff (see pyproject.toml)

```
ruff check --fix
```


## License

Wiskers has a MIT license, as found in the [LICENSE](https://github.com/vincentbonnetai/wiskers/blob/main/LICENSE) file.
