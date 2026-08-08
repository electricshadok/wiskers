from typing import Any

import streamlit as st
import torch


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


def dataset_ui(data_module: Any):
    import os

    st.markdown("#### 📂 Rollout Explorer")

    col_split, col_rollout = st.columns(2)

    with col_split:
        split = st.selectbox("Split", ["Train", "Validation", "Test"])

    if split == "Train":
        data_module.setup("fit")
        dataset = data_module.train_dataloader().dataset
    elif split == "Validation":
        data_module.setup("fit")
        dataset = data_module.val_dataloader().dataset
    else:  # Test
        data_module.setup("test")
        dataset = data_module.test_dataloader().dataset

    first = _unbatch(dataset[0])

    if not isinstance(first, dict):
        raise ValueError(f"Expected a dictionary sample, got {type(first)}")

    required_keys = {"media", "media_next", "action", "done"}
    missing_keys = required_keys - set(first.keys())
    if missing_keys:
        raise KeyError(f"Sample is missing required keys: {missing_keys}")

    # Group transitions by rollout file for sequential viewing
    rollouts = {}
    if hasattr(dataset, "transitions"):
        for idx, (fpath, step_idx) in enumerate(dataset.transitions):
            fname = os.path.basename(fpath)
            rollouts.setdefault(fname, []).append((idx, step_idx))

    with col_rollout:
        if rollouts:
            selected_rollout = st.selectbox("Select Rollout File", list(rollouts.keys()))
            rollout_steps = rollouts[selected_rollout]

            # Sort steps by step_idx
            rollout_steps = sorted(rollout_steps, key=lambda x: x[1])
        else:
            selected_rollout = None

    if selected_rollout and rollouts:
        step_selection = st.slider(
            "Step Index",
            min_value=0,
            max_value=len(rollout_steps) - 1,
            value=0,
            format="Step %d"
        )

        global_idx, step_idx = rollout_steps[step_selection]
        sample = dataset[global_idx]
    else:
        st.markdown("#### 🔍 Sample Viewer")
        idx = st.slider("Select Sample Index", 0, len(dataset) - 1, 0)
        sample = dataset[idx]

    # Display frames and action side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Observation (`media`)**")
        img = prep_image(sample["media"]).permute(1, 2, 0).numpy()
        st.image(img, width="stretch")

    with col2:
        st.markdown("**Next Observation (`media_next`)**")
        img_next = prep_image(sample["media_next"]).permute(1, 2, 0).numpy()
        st.image(img_next, width="stretch")

    # Action details
    st.markdown("#### 🕹️ Control Action & Status")
    action = sample["action"].cpu().numpy()

    # Show steering, gas, brake metrics and bars
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Steering (Left ◀ / ▶ Right)", f"{action[0]:.3f}")
        # Scale from [-1, 1] to [0, 100] for progress bar
        steer_val = float((action[0] + 1.0) / 2.0)
        st.progress(steer_val)
    with c2:
        st.metric("Gas (Accelerate)", f"{action[1]:.3f}")
        st.progress(float(action[1]))
    with c3:
        st.metric("Brake", f"{action[2]:.3f}")
        st.progress(float(action[2]))
    with c4:
        done_val = bool(sample["done"].item())
        st.metric("Done State", "True" if done_val else "False")
        if done_val:
            st.error("🚨 Episode Terminated")
        else:
            st.success("🟢 Episode Active")

