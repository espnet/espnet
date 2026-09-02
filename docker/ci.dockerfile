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

ARG PYTHON_VERSION=3.12

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

# Strip packages whose licence does not permit redistribution, and prove they
# are gone. Shipping one of these inside the image would distribute it to
# whoever can pull the image; installing it at run time, as jobs do, does not.
#
# The verification is not decoration. A silent failure here would put a licence
# violation into every pull of this image, so it fails the build instead.
#
# `python -m pip`, not `venv/bin/pip`. On python 3.13 the pip script is not there:
# setup_venv.sh pins pip==25.2, then tools/Makefile runs `ensurepip --upgrade`,
# and 3.13 bundles a pip newer than 25.2 - so unlike 3.12, that upgrade actually
# runs and leaves no bin/pip behind. This was the only place in the repository
# calling the script rather than the module; the other 36 call sites use
# `python -m pip`, which is why nothing else noticed.
RUN names=$(sed -e 's/#.*//' -e 's/[<>=!~;[].*//' -e 's/[[:space:]]//g' -e '/^$/d' ci/no_redistribute.txt) \
    && if [ -n "$names" ]; then \
         /espnet/tools/venv/bin/python -m pip uninstall -y $names \
           || { echo "ERROR: pip uninstall failed for: $names" >&2; exit 1; }; \
         for n in $names; do \
           if /espnet/tools/venv/bin/python -c "import importlib.metadata as m; m.version('$n')" >/dev/null 2>&1; then \
             echo "ERROR: $n is still installed and must not be baked into this image" >&2; \
             exit 1; \
           fi; \
         done; \
         echo "excluded from the image: $names"; \
       fi

# Non-failing: surfaces dependencies that arrive with no licence metadata, so a
# future kaldiio is noticed rather than silently shipped.
RUN /espnet/tools/venv/bin/python ci/report_unlicensed.py

# ------------------------------------------------------------------ final ----
FROM python:${PYTHON_VERSION}-bookworm

ARG TH_VERSION=2.9.1
LABEL org.opencontainers.image.source="https://github.com/espnet/espnet"
LABEL org.opencontainers.image.description="Prebuilt ESPnet CI environment (torch ${TH_VERSION})"

# The runtime half of the list above, plus everything
# .github/actions/install-system-dependencies installs, so that a job running
# in this image does not need to call apt at all. That action is currently a
# failure source in its own right: the runner image ships an apt repository at
# packages.microsoft.com that espnet installs nothing from, and when it answers
# 403 the whole job dies in `apt-get update`.
#
# The compilers stay because some tests and recipes build extensions on the
# fly; dropping them is a size optimisation to evaluate separately.
RUN apt-get update -qq \
    && apt-get install -qq -y --no-install-recommends \
        bc build-essential cmake ffmpeg git libjpeg-dev libsndfile1-dev sox \
        unzip wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /espnet/tools /espnet/tools

# What ci/install_kaldi.sh sets up, baked in. ci/install.sh removes tools/kaldi
# at the end of the build, so this is deliberately in the final stage rather
# than inherited from the builder.
#
# Doing it here rather than per job removes a git clone and a GitHub Releases
# download from every integration job - 72 of them per run, each an opportunity
# for the kind of network failure that has already cost this work several
# reruns.
COPY ci/install_kaldi.sh /tmp/install_kaldi.sh
RUN mkdir -p /espnet && cd /espnet \
    && /tmp/install_kaldi.sh \
    && rm -f /espnet/ubuntu16-featbin.tar.gz \
    && rm -rf /espnet/featbin /tmp/install_kaldi.sh \
    && test -n "$(ls -A /espnet/tools/kaldi/src/featbin/)"

# The base image puts its own interpreter first on PATH, so `pip` and `python`
# would resolve to /usr/local/bin and install into the wrong interpreter. Put
# the venv first: it is the one tools/activate_python.sh selects and the one
# the tests run under.
ENV PATH="/espnet/tools/venv/bin:${PATH}"

# The editable install baked in the builder points at a source tree that is not
# in this stage. Every job re-points it at its own checkout; nothing here
# should import espnet.
#
# A job running in this image therefore needs both:
#     pip install -e . --no-deps
#     pip install -r ci/no_redistribute.txt
WORKDIR /
