"""
Streamlit view components for exploring collected episodes, DataModule chunks, and action sweeps.
"""

import numpy as np
import streamlit as st
import torch
from tensordict import TensorDict

from webui.dataset.adapters.base import EnvUIAdapter
from webui.dataset.adapters.registry import get_adapter
from wiskers.data.datamodule import TrajectoryDataModule
from wiskers.data.episode import Episode
from wiskers.data.transforms import EpisodeToTensor
from wiskers.envs.action_spec import BoundedActionSpec


_STATUS_RENDERERS = {
    "success": st.success,
    "warning": st.warning,
    "error": st.error,
    "info": st.info,
}


def _extract_render_config(data_module: TrajectoryDataModule) -> tuple[str, tuple[float, float]]:
    """Return ``(render_key, render_range)`` from ``data_module.transform``."""
    if isinstance(data_module.transform, EpisodeToTensor) and data_module.transform.render_key is not None:
        return data_module.transform.render_key, data_module.transform.render_range
    return "render", (0.0, 1.0)


def _episode_frames(episode: Episode, obs_key: str) -> torch.Tensor:
    """Return full observation sequence ``(T + 1, ...)`` by prepending ``obs[0]`` to ``next/obs``."""
    return torch.cat([episode.obs[obs_key][:1], episode["next", "obs", obs_key]], dim=0)


def _obs_schema_dict(obs_td: TensorDict) -> dict[str, str]:
    """Build a flat dictionary describing each observation field's dtype and shape."""
    return {
        str(key): f"{str(val.dtype).replace('torch.', '')}[{', '.join(str(d) for d in val.shape)}]"
        for key, val in obs_td.items()
    }


def _obs_to_display_numpy(obs_frame: torch.Tensor, obs_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Convert either raw uint8 (H, W, C) or transformed float32 (C, H, W) tensor to numpy for ``st.image``."""
    if obs_frame.dtype == torch.uint8:
        return obs_frame.numpy()

    low, high = obs_range
    img = ((obs_frame - low) / (high - low)).clamp(0.0, 1.0)
    if img.shape[0] == 1:
        return img.squeeze(0).numpy()
    return img.permute(1, 2, 0).numpy()


def _select_step_idx(T: int, key: str) -> int:
    """Render a step slider when ``T > 1``, otherwise return step ``0``."""
    if T <= 1:
        return 0
    return st.slider(
        "Step Index",
        min_value=0,
        max_value=T - 1,
        value=0,
        format="Step %d",
        key=key,
    )


def _render_step_pair(
    obs_seq: torch.Tensor,
    step_idx: int,
    obs_range: tuple[float, float],
) -> None:
    """Render current and next observation images side by side for ``step_idx``."""
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


def _render_step_status(
    adapter: EnvUIAdapter,
    action: torch.Tensor,
    reward_val: float,
    done_val: bool,
) -> None:
    """Render formatted action, reward, and termination metrics for a single step."""
    st.markdown("#### 🕹️ Control Action & Status")
    step_info = adapter.describe_step(action, reward_val, done_val)

    cols = st.columns(len(step_info))
    for col, (label, info) in zip(cols, step_info.items()):
        with col:
            st.metric(label, info.value)
            if info.bar is not None:
                st.progress(float(info.bar))
            if info.note and info.status:
                _STATUS_RENDERERS[info.status](info.note)
            elif info.note:
                st.caption(info.note)


def _select_raw_episode(
    data_module: TrajectoryDataModule,
    obs_key: str,
    dm_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render replay-buffer summary metrics and return the selected raw episode tensors and step."""
    replay_buffer = data_module.replay_buffer
    episodes = replay_buffer.episodes
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Episodes", f"{len(episodes)} / {replay_buffer.capacity_episodes}")
    m2.metric("Total Steps", f"{replay_buffer.num_steps:,}")
    m3.metric("Avg Length", f"{sum(len(e) for e in episodes) / len(episodes):.1f}")
    render_obs = episodes[0].obs[obs_key]
    m4.metric("Obs Shape (H, W, C)", f"{tuple(render_obs.shape[1:])}")
    m5.metric("Obs Dtype", f"{render_obs.dtype} [0, 255]")

    st.markdown("**Observation Dictionary (`obs`)**")
    st.json(_obs_schema_dict(episodes[0].obs[0]))

    st.markdown("---")
    col_ep, col_step = st.columns([1, 2])
    with col_ep:
        ep_idx = st.selectbox(
            "Select Rollout Episode",
            range(len(episodes)),
            format_func=lambda i: (
                f"Episode {i + 1}  •  {len(episodes[i])} steps  •  R={episodes[i].reward.sum().item():.1f}"
            ),
            key=f"raw_ep_{dm_key}",
        )

    episode = episodes[ep_idx]
    with col_step:
        step_idx = _select_step_idx(len(episode), key=f"raw_step_{dm_key}_{ep_idx}")

    return _episode_frames(episode, obs_key), episode.action, episode.reward, episode.done, step_idx


def _select_batch_chunk(
    data_module: TrajectoryDataModule,
    obs_key: str,
    obs_range: tuple[float, float],
    dm_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render sampled-chunk summary metrics and return the selected chunk tensors and step."""
    cache_key = f"dm_batch_{dm_key}_{data_module.batch_size}_{data_module.num_steps}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = next(iter(data_module.train_dataloader()))

    batch = st.session_state[cache_key]
    num_chunks = int(batch["obs"].batch_size[0])
    episodes = data_module.replay_buffer.episodes

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
                f"R={batch['reward'][i].sum().item():.1f}"
            ),
            key=f"chunk_idx_{dm_key}",
        )

    obs_seq = batch["obs"][chunk_idx][obs_key]
    actions_seq = batch["action"][chunk_idx]
    m4.metric("Obs Shape (C, H, W)", f"{tuple(obs_seq.shape[1:])}")
    m5.metric("Obs Dtype / Range", f"{obs_seq.dtype} [{obs_range[0]:g}, {obs_range[1]:g}]")

    with col_step:
        step_idx = _select_step_idx(int(actions_seq.shape[0]), key=f"chunk_step_{dm_key}")

    return obs_seq, actions_seq, batch["reward"][chunk_idx], batch["done"][chunk_idx], step_idx


def dataset_explorer_view(data_module: TrajectoryDataModule) -> None:
    """
    Render the interactive Rollout Explorer for ``TrajectoryDataModule``,
    supporting both raw ReplayBuffer episodes and sampled/transformed DataModule chunks.
    """
    episodes = data_module.replay_buffer.episodes
    if not episodes:
        st.warning("No episodes available in replay buffer.")
        return

    dm_key = id(data_module)
    obs_key, obs_range = _extract_render_config(data_module)
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
            key=f"view_mode_{dm_key}",
        )

    if view_mode == "Replay Buffer (Raw Episodes)":
        obs_seq, actions_seq, rewards_seq, dones_seq, step_idx = _select_raw_episode(data_module, obs_key, dm_key)
    else:
        obs_seq, actions_seq, rewards_seq, dones_seq, step_idx = _select_batch_chunk(
            data_module, obs_key, obs_range, dm_key
        )

    _render_step_pair(obs_seq, step_idx, obs_range)

    action = actions_seq[step_idx]
    reward_val = float(rewards_seq[step_idx].item())
    done_val = bool(dones_seq[step_idx].item())
    _render_step_status(get_adapter(data_module), action, reward_val, done_val)

    st.markdown("#### 📈 Episode Return Curve")
    cum_rewards = rewards_seq.squeeze(-1).cumsum(dim=0).numpy()
    st.line_chart(cum_rewards, height=180)
    st.caption(
        f"Step {step_idx}: reward = **{reward_val:+.3f}**  |  "
        f"cumulative return = **{cum_rewards[step_idx]:.2f}**  |  "
        f"total return = **{cum_rewards[-1]:.2f}**"
    )


def action_sweep_view(data_module: TrajectoryDataModule, episodes: list[Episode]) -> None:
    """
    Render the single-action sweep debugger: one episode per action dimension or discrete action,
    with 10 samples across its ``ActionSpec`` bounds.
    """
    if not episodes:
        st.warning("No action sweep episodes available.")
        return

    dm_key = id(data_module)
    spec = data_module.collector.env.action_spec
    adapter = get_adapter(data_module)
    obs_key, obs_range = _extract_render_config(data_module)

    st.markdown("#### 🎛️ Single-Action Sweep")
    st.caption(
        "Each episode isolates one action from the same reset seed "
        "(linearly sweeping `[low, high]` for continuous dims or repeating action `i` for discrete specs)."
    )

    def _format_sweep_action(idx: int) -> str:
        label = adapter.action_label(idx)
        name = f"Action {idx} ({label})" if label else f"Action {idx}"
        if isinstance(spec, BoundedActionSpec):
            low_v = float(spec.low[idx].item())
            high_v = float(spec.high[idx].item())
            return f"{name}  [{low_v:g} → {high_v:g}]"
        return name

    m1, m2, m3, m4 = st.columns(4)
    st.markdown("---")
    col_act, col_step = st.columns([1, 2])

    with col_act:
        action_idx = st.selectbox(
            "Select Action to Debug",
            range(len(episodes)),
            format_func=_format_sweep_action,
            key=f"sweep_action_{dm_key}",
        )

    episode = episodes[action_idx]
    T = len(episode)
    obs_seq = _episode_frames(episode, obs_key)

    m1.metric("Action Spec", type(spec).__name__)
    m2.metric("Actions", f"{len(episodes)}")
    m3.metric("Steps / Action", f"{T}")
    if isinstance(spec, BoundedActionSpec):
        low_v = float(spec.low[action_idx].item())
        high_v = float(spec.high[action_idx].item())
        m4.metric("Sweep Range", f"[{low_v:g}, {high_v:g}]")
    else:
        lbl = adapter.action_label(action_idx)
        m4.metric("Selected Action", lbl if lbl else f"index {action_idx}")

    with col_step:
        step_idx = _select_step_idx(T, key=f"sweep_step_{dm_key}_{action_idx}")

    _render_step_pair(obs_seq, step_idx, obs_range)

    action = episode.action[step_idx]
    reward_val = float(episode.reward[step_idx].item())
    done_val = bool(episode.done[step_idx].item())

    if isinstance(spec, BoundedActionSpec):
        active_val = float(action[action_idx].item())
        formatted_action = [round(float(v), 3) for v in action.tolist()]
        st.caption(f"Active Action[{action_idx}] = **{active_val:.3f}**  |  Full Action = `{formatted_action}`")
    else:
        st.caption(f"Action = `{action.tolist()}`")

    _render_step_status(adapter, action, reward_val, done_val)
