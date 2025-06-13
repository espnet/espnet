#!/usr/bin/env bash
set -e

# Build the Docker image
echo "[INFO] Building Docker"
_root=${PWD}

docker build \
  -f "${_root}/.devcontainer/ci_cpu/espnet.dockerfile" \
  -t "espnet:dev-debian-ci" \
  --build-arg USE_IMAGE="debian:latest" \
  --build-arg USERNAME="$(whoami)" \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  "${_root}"
