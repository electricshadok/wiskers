#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="wiskers"
TAG=$(grep -m1 '^version =' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
DOCKERFILE_PATH="./Dockerfile"

docker build \
    --platform linux/amd64 \
    --build-arg "VERSION=$TAG" \
    -t "$IMAGE_NAME:$TAG" \
    -t "$IMAGE_NAME:latest" \
    -f "$DOCKERFILE_PATH" \
    .

# Push the image to a Docker registry (optional)
# docker push "$IMAGE_NAME:$TAG"
