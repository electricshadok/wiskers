import streamlit as st
import torch
import torchvision


def dataset_ui(dataset: torch.utils.data.Dataset):
    num_rows, num_cols = 10, 20
    total_images = num_rows * num_cols

    # Fetch a batch of images
    images = []
    for i in range(total_images):
        batch = dataset[i]
        image, label = batch
        image = (image + 1.0) * 0.5  # scale image to [0, 1]
        images.append(image.squeeze())

    grid = torchvision.utils.make_grid(images, nrow=num_cols, padding=2, pad_value=1)

    # Display the grid of images
    grid = grid.permute(1, 2, 0).numpy()
    st.image(grid, caption="Dataset", use_column_width=True)
