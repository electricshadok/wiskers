import streamlit as st

import streamlit_ui.dataset.ui as dataset_ui
import streamlit_ui.diffusion.ui as diffusion_ui


if __name__ == "__main__":
    tab_selection = st.sidebar.radio(
        "Select a model:",
        ("Dataset", "Diffusion"),
    )

    if tab_selection == "Dataset":
        dataset_ui.start_ui()

    elif tab_selection == "Diffusion":
        diffusion_ui.start_ui()
