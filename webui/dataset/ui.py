"""
Streamlit page orchestrator for the Wiskers Dataset Inspector.
"""

import subprocess
import sys

import streamlit as st
import torch
from hydra.utils import instantiate

from webui.dataset.adapters.registry import format_config_name
from webui.dataset.cache import (
    PROJECT_ROOT,
    apply_ui_overrides,
    get_buffer_cache_path,
    get_config_files,
    get_sweep_cache_path,
)
from webui.dataset.views import action_sweep_view, dataset_explorer_view
from wiskers.data.datamodule import TrajectoryDataModule
from wiskers.data.episode import Episode
from wiskers.pipeline.config import load_config


def run_prepare_ui(config_path: str, force: bool = False) -> None:
    """Run ``webui/prepare_ui.py`` on the main thread of a separate Python process."""
    cmd = [sys.executable, str(PROJECT_ROOT / "webui" / "prepare_ui.py"), "--config", config_path]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


@st.cache_resource(show_spinner="Loading prepared dataset...")
def load_data_module(config_path: str) -> TrajectoryDataModule:
    cache_path = get_buffer_cache_path(config_path)
    if not cache_path.exists():
        run_prepare_ui(config_path, force=False)

    config = apply_ui_overrides(load_config(config_path))
    data_module = instantiate(config.data_module, _convert_="all")
    data_module.replay_buffer = torch.load(cache_path, weights_only=False)
    try:
        data_module.setup("fit")
    finally:
        data_module.collector.close()
    return data_module


@st.cache_resource(show_spinner="Loading action sweep...")
def load_action_sweep(config_path: str) -> list[Episode]:
    cache_path = get_sweep_cache_path(config_path)
    if not cache_path.exists():
        run_prepare_ui(config_path, force=False)
    return torch.load(cache_path, weights_only=False)


def start_ui() -> None:
    st.title("🐱 Wiskers — Dataset Inspector")

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        config_path = st.selectbox(
            "Select Dataset Config",
            get_config_files(),
            format_func=format_config_name,
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Rollouts", help="Clear cached rollouts and re-seed episodes"):
            run_prepare_ui(config_path, force=True)
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["Dataset Explorer", "Action Sweep"])
    data_module = load_data_module(config_path)
    with tab1:
        dataset_explorer_view(data_module)
    with tab2:
        action_sweep_view(data_module, load_action_sweep(config_path))
