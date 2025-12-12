from typing import Any, Optional

import streamlit as st
import torch
import torchvision


def _unbatch(sample: Any) -> Any:
    if isinstance(sample, (tuple, list)) and len(sample) >= 1:
        return sample[0]
    return sample


def _is_video(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor) and x.ndim == 4  # (T,C,H,W)


def _is_image(x: torch.Tensor) -> bool:
    return isinstance(x, torch.Tensor) and x.ndim == 3  # (C,H,W)


def prep_image(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize and ensure image tensor is in (C, H, W) format with float values in [0, 1].

    Args:
        x (torch.Tensor): Input image tensor. Accepts:
            - (C, H, W) color image with dtype uint8 or float.
            - Values can be in [0, 255] (uint8), already normalized [0, 1] (float),
            or in the range [-1, 1] (float).

    Returns:
        torch.Tensor: Image tensor in (C, H, W) format, dtype float32, values scaled to [0, 1].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")

    x = x.detach().cpu()

    if x.dtype.is_floating_point:
        if x.min() < 0:  # looks like [-1,1]
            x = (x + 1.0) * 0.5
        x = x.clamp(0, 1)
    else:
        x = x.to(torch.float32) / 255.0

    return x


def dataset_ui(
    dataset: torch.utils.data.Dataset,
    num_rows: int = 6,
    num_cols: int = 10,
    max_items: Optional[int] = None,
):
    total_cells = num_rows * num_cols
    limit = max_items or total_cells
    n = min(len(dataset), limit)

    first = _unbatch(dataset[0])

    if _is_image(first):
        st.markdown("**Mode:** Image dataset")
        imgs_chw = [prep_image(_unbatch(dataset[i])) for i in range(n)]
        st.write(
            f"Sample image shape: {imgs_chw[0].shape}, "
            f"dtype: {imgs_chw[0].dtype}, "
            f"range: ({imgs_chw[0].min().item():.3f}, {imgs_chw[0].max().item():.3f})"
        )
        grid = torchvision.utils.make_grid(
            imgs_chw, nrow=num_cols, padding=2, pad_value=1.0
        )
        grid = grid.permute(1, 2, 0).numpy()  # HWC float [0,1]

        st.image(grid, caption=f"{len(imgs_chw)} images", use_container_width=True)

    elif _is_video(first):
        st.markdown("**Mode:** Video dataset")
        idx = st.number_input(
            "Select video index", min_value=0, max_value=len(dataset) - 1, value=0
        )
        video = _unbatch(dataset[idx])  # (T,C,H,W)

        T = video.shape[0]
        frame_idx = st.slider("Frame", min_value=0, max_value=T - 1, value=0, step=1)
        frame = video[frame_idx]  # (C,H,W)
        frame = prep_image(video[frame_idx])  # CHW float
        st.write(
            f"Sample image shape: {frame.shape}, "
            f"dtype: {frame.dtype}, "
            f"range: ({frame.min().item():.3f}, {frame.max().item():.3f})"
        )
        img_hwc = (frame * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        st.image(
            img_hwc,
            caption=f"Video {idx}, Frame {frame_idx}/{T-1}",
            use_container_width=True,
        )
