# Use the NVIDIA CUDA base image with version 12.3.1 and Ubuntu 20.04
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Set metadata labels for the image
LABEL maintainer="Vincent Bonnet"
LABEL version="1.0"
LABEL description="This is a custom Docker image"

# Set the working directory in the container
WORKDIR /app

# Preconfigure tzdata package for automatic timezone configuration
# Prevent the python installation to ask for geographic zone
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update package list and install Python 3
RUN apt-get update && apt-get install -y bash \
    python3.10 \
    python3-pip

# Add Python to PATH
ENV PATH="/usr/bin/python3:$PATH"

# Copy configuration
COPY configs /app/configs

# Copy package source code
COPY src /app/src

# Copy tests
COPY tests /app/tests

# Upgrade pip and install packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    numpy \
    scipy \
    tensorboard \
    omegaconf \
    pytest \
    lightning \
    einops \ 
    safetensors

# Create an alias to map python to python3
RUN echo 'alias python=python3' >> ~/.bashrc

# Add your directory to PYTHONPATH
ENV PYTHONPATH="/app/src:$PYTHONPATH"
