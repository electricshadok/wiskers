import os
from typing import List

import streamlit as st
from hydra.utils import instantiate

from webui.utils import dataset_ui
from wiskers.cli.utils import load_config


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "datasets", "breakout.yaml"),
        os.path.join("configs", "datasets", "carracing.yaml"),
    ]


@st.cache_resource(show_spinner="Collecting rollouts and preparing dataset...")
def load_data_module(config_path: str):
    config = load_config(config_path)
    data_module = instantiate(config.data_module, _convert_="all")
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
            format_func=lambda p: (
                "ALE/Breakout-v5 (breakout.yaml)"
                if "breakout" in p
                else "CarRacing-v3 (carracing.yaml)"
            ),
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Rollouts", help="Clear cached rollouts and re-seed episodes"):
            st.cache_resource.clear()
            st.rerun()

    (tab1,) = st.tabs(["Dataset Explorer"])
    with tab1:
        data_module = load_data_module(config_path)
        dataset_ui(data_module)


