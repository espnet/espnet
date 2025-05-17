ARG FROM_IMAGE=ubuntu:latest
FROM ${FROM_IMAGE}

LABEL maintainer="Nelson Yalta <nyalta21@gmail.com>"

ARG NUM_BUILD_CORES=8
ARG DOCKER_BUILT_VER
ENV DOCKER_BUILT_VER=${DOCKER_BUILT_VER}
ENV NUM_BUILD_CORES=${NUM_BUILD_CORES}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        automake \
        autotools-dev \
        bc \
        build-essential \
        cmake \
        curl \
        espeak-ng \
        ffmpeg \
        gawk \
        gfortran \
        git \
        gnupg2 \
        libffi-dev \
        libjpeg-dev \
        libtool \
        libncurses5-dev \
        nodejs \
        npm \
        pandoc \
        python3-full \
        python3-dev \
        python3-pip \
        sox \
        software-properties-common \
        sudo \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install --no-install-recommends \
        git-lfs \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    mkdir -p /workspaces

# Latest version of git
ENV TZ=Etc/UTC
ENV TH_VERSION=2.6.0
ENV USE_CONDA=false
ENV PATH=/workspaces/venv/bin:${PATH}

RUN python3 -m venv /workspaces/venv && \
    rm -rf /root/.cache/pip

WORKDIR /workspaces
