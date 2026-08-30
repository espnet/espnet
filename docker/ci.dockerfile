# Prebuilt environment for CI.
#
# Contains everything ci/install.sh produces - the venv, torch, and every make
# target - so that CI jobs do not rebuild it. A job supplies its own checkout
# and re-points the editable install with `pip install -e . --no-deps`, which
# takes seconds because every dependency is already present.
#
# Two stages on purpose: deleting the source in the same stage that copied it
# would leave it in the layer below and shrink nothing. The final stage takes
# only tools/, which is the part that took 8 minutes to build.
#
# Built by .github/workflows/build_ci_image.yml and tagged with a hash of the
# files that determine its contents.

ARG PYTHON_VERSION=3.10

# ---------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-bookworm AS builder

ARG PYTHON_VERSION
ARG TH_VERSION=2.9.1

RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends \
        automake bc build-essential cmake curl ffmpeg git libjpeg-dev \
        libsndfile1-dev libtool pkg-config sox unzip wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /espnet
WORKDIR /espnet

ENV ESPNET_PYTHON_VERSION=${PYTHON_VERSION} \
    TH_VERSION=${TH_VERSION} \
    USE_CONDA=false \
    WITH_OMP=ON

# Each cleanup tolerates its own failure. Do not collapse these into one
# `... || true` chain: that swallows a failure of ci/install.sh too, and builds
# a broken image that reports success.
RUN ./ci/install.sh \
    && { pip cache purge || true; } \
    && { find /espnet/tools -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true; }

# ------------------------------------------------------------------ final ----
FROM python:${PYTHON_VERSION}-bookworm

ARG TH_VERSION=2.9.1
LABEL org.opencontainers.image.source="https://github.com/espnet/espnet"
LABEL org.opencontainers.image.description="Prebuilt ESPnet CI environment (torch ${TH_VERSION})"

# The runtime half of the list above. The compilers stay because some tests and
# recipes build extensions on the fly; dropping them is a size optimisation to
# evaluate once the baseline measurement exists.
RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends \
        bc build-essential cmake ffmpeg git libsndfile1-dev sox unzip wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /espnet/tools /espnet/tools

# The base image puts its own interpreter first on PATH, so `pip` and `python`
# would resolve to /usr/local/bin and install into the wrong interpreter. Put
# the venv first: it is the one tools/activate_python.sh selects and the one
# the tests run under.
ENV PATH="/espnet/tools/venv/bin:${PATH}"

# The editable install baked in the builder points at a source tree that is not
# in this stage. Every job re-points it at its own checkout; nothing here
# should import espnet.
WORKDIR /
