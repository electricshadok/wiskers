#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

uv run python webui/prepare_ui.py
uv run streamlit run webui/main.py
