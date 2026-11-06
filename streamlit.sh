#!/bin/bash
set -e

python webui/prepare_ui.py
streamlit run webui/main.py

