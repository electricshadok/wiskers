import numpy as np
import streamlit as st
import torch

from webui.dataset.adapters import get_adapter
from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule


def _obs_to_display_numpy(obs_frame: torch.Tensor, obs_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Convert either raw uint8 (H, W, C) or transformed float32 (C, H, W) tensor to numpy for st.image."""
    if obs_frame.dtype == torch.uint8:
        return obs_frame.numpy()

    low, high = obs_range
    img = ((obs_frame - low) / (high - low)).clamp(0.0, 1.0)
    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()


def dataset_ui(data_module: TrajectoryDataModule) -> None:
    """
    Render the interactive Rollout Explorer for TrajectoryDataModule,
    supporting both raw ReplayBuffer episodes and sampled/transformed DataModule chunks.
    """
    replay_buffer = data_module.replay_buffer
    if replay_buffer.is_empty or data_module._dataset is None:
        data_module.setup("fit")

    episodes = list(replay_buffer._episodes)
    obs_range = data_module.transform.obs_range

    col_title, col_source = st.columns([2, 2])
    with col_title:
        st.markdown("#### 📂 Rollout Explorer")
    with col_source:
        view_mode = st.selectbox(
            "Data Source",
            [
                "Replay Buffer (Raw Episodes)",
                "DataModule (Sampled & Transformed Chunks)",
            ],
        )

    if view_mode == "Replay Buffer (Raw Episodes)":
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Episodes", f"{len(episodes)} / {replay_buffer.capacity_episodes}")
        m2.metric("Total Steps", f"{replay_buffer.num_steps:,}")
        m3.metric("Avg Length", f"{sum(len(e) for e in episodes) / len(episodes):.1f}")
        m4.metric("Obs Shape (H, W, C)", f"{tuple(episodes[0].obs.shape[1:])}")
        m5.metric("Obs Dtype", f"{episodes[0].obs.dtype} [0, 255]")

        st.markdown("---")
        col_ep, col_step = st.columns([1, 2])

        with col_ep:
            ep_idx = st.selectbox(
                "Select Rollout Episode",
                range(len(episodes)),
                format_func=lambda i: (
                    f"Episode {i + 1}  •  {len(episodes[i])} steps  •  "
                    f"R={episodes[i].rewards.sum().item():.1f}"
                ),
            )

        episode = episodes[ep_idx]
        obs_seq, actions_seq, rewards_seq, dones_seq = (
            episode.obs,
            episode.actions,
            episode.rewards,
            episode.dones,
        )
    else:
        cache_key = f"dm_batch_{id(data_module)}_{data_module.batch_size}_{data_module.num_steps}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = next(iter(data_module.train_dataloader()))

        batch = st.session_state[cache_key]
        num_chunks = int(batch["obs"].shape[0])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Batch Size", f"{num_chunks}")
        m2.metric("Chunk Steps (T)", f"{data_module.num_steps}")
        m3.metric("Buffer Episodes", f"{len(episodes)}")

        st.markdown("---")
        col_ep, col_step = st.columns([1, 2])

        with col_ep:
            chunk_idx = st.selectbox(
                "Select Batch Chunk",
                range(num_chunks),
                format_func=lambda i: (
                    f"Chunk {i + 1} / {num_chunks}  •  {data_module.num_steps} steps  •  "
                    f"R={batch['rewards'][i].sum().item():.1f}"
                ),
            )

        obs_seq = batch["obs"][chunk_idx]
        actions_seq = batch["actions"][chunk_idx]
        rewards_seq = batch["rewards"][chunk_idx]
        dones_seq = batch["dones"][chunk_idx]

        m4.metric("Obs Shape (C, H, W)", f"{tuple(obs_seq.shape[1:])}")
        m5.metric("Obs Dtype / Range", f"{obs_seq.dtype} [{obs_range[0]:g}, {obs_range[1]:g}]")

    T = int(actions_seq.shape[0])
    with col_step:
        step_idx = st.slider(
            "Step Index",
            min_value=0,
            max_value=T - 1,
            value=0,
            format="Step %d",
        )

    col1, col2 = st.columns(2)
    obs_curr = obs_seq[step_idx]
    obs_next = obs_seq[step_idx + 1]

    with col1:
        st.markdown(
            f"**Current Observation (`obs[{step_idx}]`, shape `{tuple(obs_curr.shape)}`, "
            f"dtype=`{obs_curr.dtype}`, min=`{obs_curr.min().item():.2f}`, max=`{obs_curr.max().item():.2f}`)**"
        )
        st.image(_obs_to_display_numpy(obs_curr, obs_range), width="stretch")

    with col2:
        st.markdown(
            f"**Next Observation (`obs[{step_idx + 1}]`, shape `{tuple(obs_next.shape)}`, "
            f"dtype=`{obs_next.dtype}`, min=`{obs_next.min().item():.2f}`, max=`{obs_next.max().item():.2f}`)**"
        )
        st.image(_obs_to_display_numpy(obs_next, obs_range), width="stretch")

    # ── Action & Status ──────────────────────────────────────────────────
    st.markdown("#### 🕹️ Control Action & Status")
    action = actions_seq[step_idx]
    reward_val = float(rewards_seq[step_idx].item())
    done_val = bool(dones_seq[step_idx].item())

    adapter = get_adapter(data_module)
    step_info = adapter.describe_step(action, reward_val, done_val)

    cols = st.columns(len(step_info))
    for col, (label, info) in zip(cols, step_info.items()):
        with col:
            st.metric(label, info["value"])
            if info.get("bar") is not None:
                st.progress(float(info["bar"]))
            note = info.get("note")
            status = info.get("status")
            if note and status:
                getattr(st, status)(note)
            elif note:
                st.caption(note)

    # ── Return trajectory curve ──────────────────────────────────────────
    st.markdown("#### 📈 Episode Return Curve")
    cum_rewards = rewards_seq.cumsum(dim=0).numpy()
    st.line_chart(cum_rewards, height=180)
    st.caption(
        f"Step {step_idx}: reward = **{reward_val:+.3f}**  |  "
        f"cumulative return = **{cum_rewards[step_idx]:.2f}**  |  "
        f"total return = **{cum_rewards[-1]:.2f}**"
    )
