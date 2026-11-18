# NVIDIA CUDA base image (Ubuntu 22.04)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG VERSION=0.1.0

# Image metadata
LABEL maintainer="Vincent Bonnet"
LABEL version="${VERSION}"
LABEL description="Docker image for Wiskers"

WORKDIR /app

# Configure timezone non-interactively
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System dependencies for swig/Box2D compilation and headless MuJoCo/OpenGL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    libegl1 \
    libgl1 \
    libgles2 \
    libglfw3 \
    libglvnd0 \
    libx11-6 \
    libxext6 \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MUJOCO_GL=egl \
    PATH="/app/.venv/bin:$PATH"

# Install Python 3.13 and third-party dependencies (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv python install 3.13 && \
    uv sync --locked --no-install-project --no-dev

# Copy project files and install wiskers
COPY README.md ./
COPY wiskers ./wiskers
COPY configs ./configs
COPY webui ./webui
COPY tests ./tests
RUN uv sync --locked --no-dev
