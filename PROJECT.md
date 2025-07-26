# Wiskers: A Generative Physics Reasoning System

**Wiskers** combines **physics-aware video understanding**, **causal reasoning**, and **text-driven scene editing** to enable intelligent agents that can interpret, imagine, and alter physical events.

It allows agents to analyze, edit, and generate physically grounded videos using latent models. The system supports **counterfactual simulation**, **inference-time reasoning**, and **objective-driven planning**.

---

## Core Capabilities

- Understand physical events in videos (e.g., collisions, falls, interactions)
- Answer causal and temporal questions (e.g., “What caused the box to fall?”)
- Simulate counterfactuals (e.g., “What if the ramp was removed?”)
- Generate and edit videos from text prompts using latent generative models

---

## Blueprint for Physical Reasoning Systems

Wiskers is guided by principles for building **reasoning-capable, physics-aware generative systems**:

- Learning should occur in **latent spaces** that support compact and editable representations  
- Reasoning should happen **at inference time**, not just training — meaning **computation should scale with task complexity**
- **Autoregressive or iterative inference** may offer better temporal reasoning than fixed-depth transformers  
- Systems should learn **world models from diverse sensory data** (e.g., vision, motion, touch)  
- Planning should be framed as **objective-driven inference**, incorporating constraints like safety and stability  
- Reasoning and generation can be unified under **differentiable optimization frameworks** (e.g., energy-based inference, trajectory planning, goal-conditioned generation)

---

## Implemented Features

- Multi-head and scaled dot-product attention (1D, 2D, 3D)
- CBAM and squeeze-and-excitation attention modules
- Non-local blocks for video reasoning
- Causal, separable, and residual convolution blocks
- Spatial upsampling and downsampling
- Sinusoidal positional embeddings
- Adaptive Instance Normalization (AdaIN)
- DDPIM denoising scheduler
- Beta scheduler options: linear, cosine, quadratic, sigmoid
- Model export support: checkpoints, `safetensors`, ONNX
