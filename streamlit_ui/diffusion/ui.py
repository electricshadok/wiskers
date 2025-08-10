import os
from typing import List

import streamlit as st

from streamlit_ui.diffusion.noise_debug_ui import noise_debug_ui


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "diffusion", "train.yaml"),
    ]


def start_ui():
    st.title("Wiskers(diffusion) debugging app")

    config_path = st.selectbox("Configs", get_config_files())

    noise_debug_ui(config_path)
