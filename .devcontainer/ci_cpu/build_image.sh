#!/usr/bin/env bash
set -e

# Build the Docker image
echo "[INFO] Building Docker"
_root=${PWD}

if [ ! -f ".devcontainer/container.env" ]; then
  cp .devcontainer/container.env.example .devcontainer/container.env
fi

docker build \
  -f "${_root}/.devcontainer/ci_cpu/espnet.dockerfile" \
  -t "espnet/dev:ci" \
  --build-arg USERNAME="$(whoami)" \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  "${_root}"
