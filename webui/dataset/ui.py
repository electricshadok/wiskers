import subprocess
import sys
from pathlib import Path

import streamlit as st
import torch
from hydra.utils import instantiate

from webui.prepare_ui import (
    PROJECT_ROOT,
    apply_ui_overrides,
    get_cache_path,
    get_config_files,
)
from webui.utils import dataset_ui
from wiskers.cli.utils import load_config


def format_config_name(path: str) -> str:
    """Format config file paths into descriptive display labels."""
    stem = Path(path).stem.lower()
    display_names = {
        "breakout": "ALE/Breakout-v5",
        "carracing": "CarRacing-v3",
        "franka": "Franka Kitchen",
    }
    label = display_names.get(stem, stem.replace("_", " ").title())
    return f"{label} ({Path(path).name})"


def run_prepare_ui(config_path: str, force: bool = False) -> None:
    """Run webui/prepare_ui.py on the main thread of a separate Python command."""
    cmd = [sys.executable, str(PROJECT_ROOT / "webui" / "prepare_ui.py"), "--config", config_path]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


@st.cache_resource(show_spinner="Loading prepared dataset...")
def load_data_module(config_path: str):
    cache_path = get_cache_path(config_path)
    if not cache_path.exists():
        run_prepare_ui(config_path, force=False)

    config = apply_ui_overrides(load_config(config_path))
    data_module = instantiate(config.data_module, _convert_="all")
    data_module.replay_buffer = torch.load(cache_path, weights_only=False)
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module


def start_ui():
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
            st.rerun()

    (tab1,) = st.tabs(["Dataset Explorer"])
    with tab1:
        data_module = load_data_module(config_path)
        dataset_ui(data_module)
