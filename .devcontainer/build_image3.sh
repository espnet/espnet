#!/usr/bin/env bash
set -e

processor=$1
_root=${PWD}

if [ ! -f ".devcontainer/container.env" ]; then
  touch .devcontainer/container.env
fi

# Build the Docker image
echo "[INFO] Building Docker for ${processor}"

docker build \
    -f "${_root}/.devcontainer/espnet3.dockerfile" \
    -t "espnet3:dev-${processor}" \
    --build-arg FROM_TAG=dev3-${processor} \
    --build-arg USERNAME="$(whoami)" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    "${_root}"

# source .devcontainer/container.env

# if [[ -z "${EXPORT_DATA}+x" ]]; then
#   export EXPORT_DATA=${_root}
# fi
