import os
from typing import List

import streamlit as st

from streamlit_ui.utils import dataset_ui
from wiskers.common.commands.utils import load_config
from wiskers.utils import get_data_module


def get_config_files() -> List[str]:
    return [
        os.path.join("configs", "datasets", "cifar10.yaml"),
        os.path.join("configs", "datasets", "clevrer.yaml"),
    ]


def start_ui():
    st.title("Wiskers(dataset) debugging app")

    config_path = st.selectbox("Configs", get_config_files())

    (tab1,) = st.tabs(["Dataset"])
    with tab1:
        config = load_config(config_path)
        data_module = get_data_module(
            config.data_module_type,
            **config.data_module,
        )
        data_module.prepare_data()
        data_module.setup("fit")
        dataloader = data_module.train_dataloader()
        dataset_ui(dataloader.dataset)
