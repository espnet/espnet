# Run a published ESPnet model with one command, without installing anything:
#
#   docker run --rm -v "$PWD:/data" \
#       -v "$HOME/.cache/huggingface:/cache/huggingface" \
#       espnet/espnet:inference-cpu-latest asr /data/audio.wav
#
#   docker run --rm --gpus all -v "$PWD:/data" \
#       -v "$HOME/.cache/huggingface:/cache/huggingface" \
#       espnet/espnet:inference-gpu-latest asr /data/audio.wav --device cuda
#
# One dockerfile builds both: TORCH_INDEX_URL decides which torch goes in, and
# nothing else differs. The CUDA wheels carry the CUDA runtime themselves, so
# the GPU image needs no CUDA base image - only a host driver new enough for
# the toolkit the wheels were built against.
#
# This is not espnet.dockerfile's smaller sibling - it is a different image for
# a different reader. The cpu/gpu images carry Kaldi, the recipe tooling and the
# training stack because they exist to run egs2; this one carries what
# `pip install espnet` installs (the inference dependency set, no [train]
# extra), so that trying a model is a download rather than an afternoon.

ARG PYTHON_VERSION=3.12


FROM python:${PYTHON_VERSION}-slim AS builder
LABEL maintainer="ESPnet developers <espnet@googlegroups.com>"

# Every inference dependency publishes a manylinux wheel today, but a
# transitive sdist would need a compiler at install time and there is no way to
# find that out except by building. The toolchain is therefore here, in a stage
# nothing is copied out of except the virtualenv, rather than in the image the
# user pulls.
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# A virtualenv rather than the system python: it is one directory to copy into
# the final stage, which leaves the build's apt and pip debris behind.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# Which torch, and therefore which image: the +cpu build is the same torch
# without the ~3 GB of nvidia-* wheels the PyPI one bundles, and a +cuXXX build
# carries the CUDA runtime it needs. Install it before espnet so that espnet's
# `torch>=...` is already satisfied and pip does not resolve back to PyPI.
# TH_VERSION is the version ci/image_variants.json builds and
# tools/installers/install_torch.sh installs; ci/check_ci_image_config.py fails
# if it drifts off that set.
ARG TH_VERSION=2.11.0
# torchaudio is pinned separately because the two stopped moving together:
# torchaudio's last release is 2.11.0, and every supported torch above it pairs
# with that one version (see the dependency comment in pyproject.toml).
ARG TORCHAUDIO_VERSION=2.11.0
# https://download.pytorch.org/whl/cu126 for the GPU image; CUDA 12.6 is what
# the gpu development image builds against too.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" \
    "torch==${TH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

# From PyPI, at a release, not from this checkout: the image is meant to be the
# published package, so that what it runs is what `pip install espnet` gives.
# The default tracks version.txt; the publish workflow passes it explicitly.
ARG ESPNET_VERSION=202610
RUN pip install "espnet==${ESPNET_VERSION}"


FROM python:${PYTHON_VERSION}-slim
LABEL maintainer="ESPnet developers <espnet@googlegroups.com>"
LABEL description="Run a published ESPnet model from the command line."

# The virtualenv records the absolute path of the interpreter that made it, so
# both stages have to be the same python:<version>-slim - which is what the one
# shared PYTHON_VERSION above is for.
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# No model weights are baked in. One OWSM checkpoint is about 4 GB, the
# subcommands' defaults are three different checkpoints between them, and any
# Hugging Face tag can be asked for with --model, so an image holding them would
# be both enormous and wrong for most users. They download on first use into the
# cache below; mount it (-v "$HOME/.cache/huggingface:/cache/huggingface") or
# every `docker run` downloads the model again.
#
# /cache rather than /root/.cache: on a Linux host, a container writing to a
# bind mount as root leaves root-owned files behind in the user's directory, so
# the answer is `--user "$(id -u):$(id -g)"` - and an arbitrary uid has no home
# directory in this image and cannot write under /root. Everything the process
# writes therefore lives in one world-writable directory (sticky, like /tmp)
# that is the same path whoever runs it.
#   HOME     espnet_model_zoo caches a tag published outside the Hub under
#            ~/.cache/espnet_model_zoo rather than through huggingface_hub.
#   HF_HOME  what huggingface_hub reads; a Hub tag lands here.
#   NUMBA    librosa's JIT functions are @jit(cache=True), and their cache would
#            otherwise be written beside the installed librosa, which is not
#            writable except as root.
ENV HOME=/cache \
    HF_HOME=/cache/huggingface \
    NUMBA_CACHE_DIR=/cache/numba
RUN mkdir -p /cache/huggingface /cache/numba && chmod -R 1777 /cache

# soundfile and sentencepiece ship their native libraries inside their wheels,
# so the final image needs no libsndfile or build tools of its own.

# Relative paths in `docker run ... asr audio.wav` should mean the mounted
# directory, which is what a user who typed -v "$PWD:/data" expects.
WORKDIR /data

ENTRYPOINT ["espnet"]
# `docker run espnet/espnet:inference-cpu-latest` with no arguments should say
# what the image does rather than fail on a missing subcommand.
CMD ["--help"]
