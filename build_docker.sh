#!/bin/bash

# Set variables
IMAGE_NAME="wiskers"
TAG="1.0"
DOCKERFILE_PATH="./Dockerfile"

# Build the Docker image
docker build -t "$IMAGE_NAME:$TAG" -f "$DOCKERFILE_PATH" .

# Push the image to a Docker registry (optional)
# docker push "$IMAGE_NAME:$TAG"
