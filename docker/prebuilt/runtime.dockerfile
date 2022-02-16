ARG FROM_TAG
ARG NUM_BUILD_CORES=8
ARG DOCKER_VER

FROM ubuntu:${FROM_TAG} AS main_builder
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ENV DOCKER_BUILT_VER ${DOCKER_VER}
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get -y install --no-install-recommends \ 
        automake \
        autoconf \
        apt-utils \
        bc \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flac \
        ffmpeg \
        gawk \
        gfortran \
        git \
        libboost-all-dev \
        libtool \
        libbz2-dev \
        liblzma-dev \
        libsndfile1-dev \
        patch \
        python2.7 \
        python3 \
        software-properties-common \
        sox \
        subversion \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Latest version of git
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt update && \
    apt install -y --no-install-recommends git-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi /opt/kaldi

RUN wget --tries=3 -nv "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

WORKDIR /

FROM main_builder AS espnet1
# # Using kaldi pre-built binaries
RUN cd /opt/kaldi/tools &&  \
    echo "" > extras/check_dependencies.sh && \
    chmod +x extras/check_dependencies.sh &&  \
    cd /opt/kaldi && \
    wget --tries=3 -nv https://github.com/espnet/kaldi-bin/releases/download/v0.0.1/ubuntu16-featbin.tar.gz && \
    tar -xf ./ubuntu16-featbin.tar.gz && \
    cp featbin/* src/featbin/ && \
    rm -rf featbin && \
    rm -f ubuntu16-featbin.tar.gz

WORKDIR /
