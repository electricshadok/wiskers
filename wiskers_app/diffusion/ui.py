import os
from typing import List

import streamlit as st

from wiskers_app.common.dataset_ui import dataset_ui
from wiskers_app.diffusion.noise_debug_ui import noise_debug_ui
from wiskers_app.diffusion.utils import get_dataset


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "diffusion", "default", "train.yaml"),
        os.path.join("configs", "diffusion", "debug", "train.yaml"),
        os.path.join("wiskers", "diffusion", "commands", "train.yaml"),
    ]


def start_ui():
    st.title("Wiskers(diffusion) debugging app")

    config_path = st.selectbox("Configs", get_config_files())

    tab1, tab2 = st.tabs(["Noise Debug", "Dataset"])
    with tab1:
        noise_debug_ui(config_path)

    with tab2:
        dataset_ui(get_dataset(config_path))
