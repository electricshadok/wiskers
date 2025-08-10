import os
from typing import List

import streamlit_ui as st
from streamlit_ui.common.dataset_ui import dataset_ui
from streamlit_ui.vae.utils import get_dataset


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "vae", "train.yaml"),
        os.path.join("configs", "vae", "train_debug.yaml"),
    ]


def start_ui():
    st.title("Wiskers(vae) debugging app")

    config_path = st.selectbox("Configs", get_config_files())

    (tab1,) = st.tabs(["Dataset"])
    with tab1:
        dataset_ui(get_dataset(config_path))
