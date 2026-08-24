import os
from typing import List

import streamlit as st
from hydra.utils import instantiate

from webui.utils import dataset_ui
from wiskers.cli.utils import load_config


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "datasets", "carracing.yaml"),
        os.path.join("configs", "datasets", "breakout.yaml"),
    ]


def start_ui():
    st.title("🐱 Wiskers — Dataset Inspector")

    config_path = st.selectbox("Select Dataset Config", get_config_files())

    (tab1,) = st.tabs(["Dataset Explorer"])
    with tab1:
        config = load_config(config_path)
        data_module = instantiate(config.data_module, _convert_="all")
        data_module.prepare_data()
        dataset_ui(data_module)

