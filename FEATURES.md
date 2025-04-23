## Features

**Common**
- Attention mechanisms 
    - for audio/sequences (1D): multihead, scaled-dot-product 
    - for images (2D): CBAM, squeeze-and-excitation, self-multihead, self-scaled-dot-product
    - for videos (3D): non-local-block
- Convolution Blocks
    - for audio/sequences (1D): causal convolution
    - for images (2D): separable convolution, residual block, spatial downsampling/usampling block
- Sinusoidal position embedding
- AdaIN normalization
- CIFAR10 dataset
- Support model formats (checkpoint, safetensor, onnx)

**Diffusion**
- [x] Lightning module and logging to track loss and noise stats.
- [x] DDPIM scheduler
- [x] Beta schedulers (linear, cosine, quadratic, sigmoid)

**VAE**
- [x] Lightning module and logging to track loss
- [x] Autoencoder and VAE


More details on future developments on the [TODO](https://github.com/vincentbonnetai/wiskers/blob/main/TODO.md) list.