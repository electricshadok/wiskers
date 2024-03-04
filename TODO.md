# TODO List

This todo list is continually evolving ! 


## Reusable Architectures
Evaluate different attention mechanisms to improve quality or performance. ([Visual Attention survey](https://arxiv.org/abs/2204.07756))
- [ ] in MHA and scaleddot attention : combine KQV into a single tensor for optimization
- [ ] seblock and cbam are implemented differently. Use conv2d instead of linear
- [ ] evaluate MemEffAttention ([code example](https://github.com/yenchenlin/dinov2-adaLN/blob/main/dinov2/layers/attention.py))
- [ ] evalutate xformers ([Hugging Face - XFormers](https://huggingface.co/docs/diffusers/optimization/xformers))
- [ ] add multi-Scale Attention ([arxiv](https://arxiv.org/abs/2103.06104))
- [ ] add AttentionUnet ([arxiv](https://arxiv.org/pdf/1804.03999.pdf))
- [ ] add PixelShuffle/PixelUnshuffle for super resolution
- [ ] add BAM (bottleneck attention module)
- [ ] remove skip_x in AttnUpBlock2D and UpBlock2D from common (should be usable between unet/ae)
- [ ] add the differents modes in non-local-block


## Training improvement
- [ ] add more schedulers ([DDPM](https://arxiv.org/pdf/2010.02502.pdf), [ImprovedDDPM](https://arxiv.org/pdf/2102.09672.pdf), [PDNM](https://arxiv.org/pdf/2202.09778.pdf))
- [ ] add v-prediction
- [ ] add Exponential Moving Average (EMA) - see DDPM paper
- [ ] add classifier-free guidance ([paper](https://openreview.net/pdf?id=qw8AKxfYbI))
- [ ] add weight initialization on VAE, diffusion models


## Add Conditioning and Latent Nodel
- [ ] add latent diffusion models.
- [ ] add LoRA in the diffusion model
- [ ] add class-conditional (one-hot encoding ?)
- [ ] add condition on text
- [ ] add condition images (e.g. ControlNet)


## Transformer-Based diffusion
Experiment with Transformer-based model.
- [ ] develop a patch embedding (Read dino code [code example](https://github.com/yenchenlin/dinov2-adaLN/blob/main/dinov2/layers/))
- [ ] add simple transformer-based diffusion model
- [ ] implement wiskers/common/models/vit_2d.py


## Sequence modeling 
- [ ] Add CausalConv2d
- [ ] Add CausalConv3d
- [ ] Add TCN


## Variational Auto-Encoder
- [ ] implement cvae
- [ ] implement vqvae
- [ ] implement beta-vae
- [ ] implement nvae :  deep hierarchical variational autoencoder


## Architectures with Temporal Data
- [ ] (2+1)D Convolution : capture both spatial (2D) and temporal (1D) features instead of


## Production, Benchmark and Debugging
Improve automation and scale training.
- [ ] add datasets (synthetic data with [gym](https://gymnasium.farama.org/), [huggingface datasets](https://huggingface.co/datasets))
- [ ] add stats under each images (min,mean,max) in streamlit app
- [ ] add FID under each images in streamlit app
- [ ] add pytest and coverage in github actions
- [ ] run code on GPU cloud service (paperspace)
- [ ] visualize trace.json from pytorch profiler into tensorboard
- [ ] benchmark script should detect max memory consumption at runtime

## Inference and Delivery
After training, inference should run outside the development environment
- [ ] run inference from onnx or safetensor files
- [ ] complete ONNXInference in generate.py script
- [ ] complete SafeTensorInference in generate.py script
- [ ] add FID metric in generate.py script


## Beyond Diffusion 
Wiskers is designed to support generative models beyond diffusion
- [ ] add wiskers/vae (vae, vqvae ... )
- [ ] add wiskers/normalizing-flow
- [ ] add wiskers/gan
- [ ] add wiskers/ebm
- [ ] add wiskers/autoregressive

