import streamlit as st
import streamlit_ui.diffusion.ui as diffusion_ui
import streamlit_ui.vae.ui as vae_ui


if __name__ == "__main__":
    tab_selection = st.sidebar.radio(
        "Select a model:",
        ("Diffusion", "VAE"),
    )

    if tab_selection == "Diffusion":
        diffusion_ui.start_ui()
    elif tab_selection == "VAE":
        vae_ui.start_ui()
