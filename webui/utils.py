from typing import Any

import streamlit as st
import torch

from webui.dataset.adapters import get_adapter


def dataset_ui(data_module: Any):
    """
    Render the interactive Rollout Explorer for trajectory-based DataModules
    (e.g., BreakoutDataModule, CarRacingDataModule).
    """
    replay_buffer = getattr(data_module, "replay_buffer", None)
    if replay_buffer is None:
        st.error(f"Expected DataModule with a ReplayBuffer, got {type(data_module).__name__}")
        return

    if replay_buffer.is_empty:
        data_module.setup("fit")

    episodes = list(replay_buffer._episodes)
    if not episodes:
        st.warning("No episodes found in replay buffer.")
        return

    st.markdown("#### 📂 Rollout Explorer")

    # Buffer Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Episodes", f"{len(episodes)} / {replay_buffer.capacity_episodes}")
    m2.metric("Total Steps", f"{replay_buffer.num_steps:,}")
    m3.metric("Chunk Length", f"{data_module.chunk_len}")
    avg_len = sum(len(e) for e in episodes) / len(episodes)
    m4.metric("Avg Episode Length", f"{avg_len:.1f}")

    st.markdown("---")
    col_ep, col_step = st.columns([1, 2])

    with col_ep:
        ep_idx = st.selectbox(
            "Select Rollout Episode",
            range(len(episodes)),
            format_func=lambda i: (
                f"Episode {i + 1}  •  {len(episodes[i])} steps  •  "
                f"R={episodes[i].rewards.sum():.1f}"
            ),
        )

    episode = episodes[ep_idx]
    T = len(episode)

    with col_step:
        step_idx = st.slider(
            "Step Index",
            min_value=0,
            max_value=T - 1,
            value=0,
            format="Step %d",
        )

    # Observations: current and next
    col1, col2 = st.columns(2)

    obs_curr = episode.obs[step_idx]
    obs_next = episode.obs[step_idx + 1] if step_idx + 1 < T else episode.obs[step_idx]

    with col1:
        st.markdown(f"**Current Observation (`obs[{step_idx}]`)**")
        st.image(obs_curr, use_container_width=True)

    with col2:
        next_label = f"obs[{step_idx + 1}]" if step_idx + 1 < T else f"obs[{step_idx}] (terminal)"
        st.markdown(f"**Next Observation (`{next_label}`)**")
        st.image(obs_next, use_container_width=True)

    # ── Action & Status ──────────────────────────────────────────────────
    st.markdown("#### 🕹️ Control Action & Status")
    action = torch.from_numpy(episode.actions[step_idx])
    reward_val = float(episode.rewards[step_idx])
    done_val = bool(episode.dones[step_idx])

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
    cum_rewards = episode.rewards.cumsum()
    st.line_chart(cum_rewards, height=180)
    st.caption(
        f"Step {step_idx}: reward = **{reward_val:+.3f}**  |  "
        f"cumulative return = **{cum_rewards[step_idx]:.2f}**  |  "
        f"episode total return = **{cum_rewards[-1]:.2f}**"
    )
