# Wiskers: DL and  Generative AI Study

This repository serves as a learning hub for experimenting with foundational deep learning (DL) architectures and Generative AI. Currently, I am exploring a variety of deep learning techniques (diffusion models, VAEs) to generate images, videos, audio, and animations into a single, cohesive resource. Expect this repo to continuously evolve.

Fair warning: My curiosity may lead this repo down multiple paths, but fear not—a windy journey is just another step towards clarity. Let’s have fun!


## Features

**Common**
- [x] Attention mechanisms 
    - for audio/sequences (1D): multihead, scaled-dot-product 
    - for images (2D): CBAM, squeeze-and-excitation, self-multihead, self-scaled-dot-product
    - for videos (3D): non-local-block
- [x] Convolution Blocks
    - for audio/sequences (1D): causal convolution
    - for images (2D): separable convolution, residual block, spatial downsampling/usampling block
- [x] Sinusoidal position embedding
- [x] AdaIN normalization
- [x] CIFAR10 dataset
- [x] Support model formats (checkpoint, safetensor, onnx)

**Diffusion**
- [x] Lightning module and logging to track loss and noise stats.
- [x] DDPIM scheduler
- [x] Beta schedulers (linear, cosine, quadratic, sigmoid)

**VAE**
- [x] Lightning module and logging to track loss
- [x] Autoencoder and VAE


More details on future developments on the [TODO](https://github.com/vincentbonnetai/wiskers/blob/main/TODO.md) list.

## Training

Use the following command to begin the training process. The configuration is defined in *train.yaml*, which can be tailored to suit your specific training needs.

<details>
<summary>Examples</summary>

**Diffusion**
```
python wiskers/diffusion/commands/train.py --config configs/diffusion/default/train.yaml
```

**VAE**
```
python wiskers/vae/commands/train.py --config configs/vae/default/train.yaml
```

</details>

## Generate Samples

Use trained model to generate samples. This command triggers the sample generation proces with the configuration *generate.yaml*.

<details>
<summary>Examples</summary>

**Diffusion**
```
python wiskers/diffusion/commands/generate.py --config configs/diffusion/default/generate.yaml
```

**VAE**
```
python wiskers/vae/commands/generate.py --config configs/vae/default/generate.yaml
```
</details>

## Streamlit Development App

Streamlit is used to debug and run diffusion code, facilitating a smoother development and testing process.

```
streamlit run wiskers_app/main.py
```

<details>
<summary>Screenshot</summary>
<p align="center"><img src="docs/app.png?raw=true"></p>
</details>

## Benchmark

More about benchmarks [HERE](https://github.com/vincentbonnetai/wiskers/blob/main/benchmarks/README.md) 

## Tests

Wiskers includes a suite of tests and a debug pipeline for this purpose.

### run debugging pipeline

Ideal for developmental checks, the debug pipeline runs a condensed version of the training process. Use the configuration in *debug/train.yaml* for this purpose.

<details>
<summary>Examples</summary>

**Diffusion**
```
python wiskers/diffusion/commands/train.py --config configs/diffusion/debug/train.yaml
```

**VAE**
```
python wiskers/vae/commands/train.py --config configs/vae/debug/train.yaml
```
</details>

### run tests

To maintain code quality and functionality, run the provided tests using pytest.

```
pytest tests
```


## Docker

### build docker container

Docker ensures a consistent environment Docker. Simply run the command to build the Docker container:

```
./build.sh
```

### start docker container
Once the build is complete, you can start the Docker container with all necessary dependencies installed:

```
docker run --rm -it diffusion:1.0
```

## License

Wiskers has a MIT license, as found in the [LICENSE](https://github.com/vincentbonnetai/wiskers/blob/main/LICENSE) file.
