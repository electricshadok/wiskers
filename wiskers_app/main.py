import streamlit as st

import wiskers_app.diffusion.ui as diffusion_ui
import wiskers_app.vae.ui as vae_ui


if __name__ == "__main__":
    tab_selection = st.sidebar.radio(
        "Select a model:",
        ("Diffusion", "VAE"),
    )

    if tab_selection == "Diffusion":
        diffusion_ui.start_ui()
    elif tab_selection == "VAE":
        vae_ui.start_ui()
