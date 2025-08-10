import torch
import torchvision

import streamlit_ui as st
from streamlit_ui.diffusion.utils import get_dataset, get_diffuser_module
from wiskers.common.commands.utils import load_config


def noise_debug_ui(config_path: str):
    config = load_config(config_path)

    st.write("### Variance Scheduler")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        config.module.scheduler.beta_start = st.number_input("Beta start", value=1e-5, format="%0.5f")

    with col2:
        config.module.scheduler.beta_end = st.number_input("Beta end", value=2e-2, format="%0.5f")

    with col3:
        config.module.scheduler.num_steps = st.number_input("Num steps", value=100, step=1)

    with col4:
        config.module.scheduler.beta_schedule = st.selectbox(
            "Select a variance type", ["linear", "quadratic", "sigmoid", "cosine"]
        )

    diffuser_module = get_diffuser_module(config)
    scheduler = diffuser_module.scheduler

    st.line_chart(scheduler.betas.numpy(), height=200)

    st.write(f"### Noising Process ({config.module.scheduler.beta_schedule})")

    dataset = get_dataset(config_path)
    max_index = len(dataset) - 1
    index = st.number_input(f"Sample index (0-{max_index})", min_value=0, max_value=max_index, step=1)
    image, label = dataset[index]

    # Generate noise with scheduler
    num_images = 10
    images = image.unsqueeze(0).repeat(num_images, 1, 1, 1)  # [N, C, W, H]
    t = torch.linspace(0, scheduler.num_steps() - 1, steps=num_images, dtype=torch.long, device=images.device)
    noise = torch.randn_like(images)
    noisy_images = scheduler.q_sample(images, t, noise)

    # Turn into images
    noisy_images = (noisy_images + 1.0) * 0.5  # [-1,1] to [0, 1]
    noisy_images = noisy_images.clip(min=0, max=1)
    grid = torchvision.utils.make_grid(noisy_images, nrow=len(noisy_images), padding=2, pad_value=1)
    grid = grid.permute(1, 2, 0).numpy()
    st.image(grid, caption=f"idx: {index}, label{label}", use_column_width=True)

    st.write("### FID")
    st.write("todo")
