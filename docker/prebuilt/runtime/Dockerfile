ARG FROM_TAG
FROM ubuntu:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG DOCKER_VER
ENV DOCKER_BUILT_VER ${DOCKER_VER}}

ARG NUM_BUILD_CORES=8
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install --no-install-recommends \ 
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
        libtool \
        libsndfile1-dev \
        patch \
        python2.7 \
        python3 \
        sox \
        subversion \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Kaldi
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi /opt/kaldi

WORKDIR /opt/kaldi/tools

RUN ./extras/install_mkl.sh -sp debian intel-mkl-64bit-2019.2-057 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    make all && \
    rm -r openfst-*/src && \
    ./extras/install_beamformit.sh && \
    ./extras/install_irstlm.sh

WORKDIR /kaldi/src 

RUN  ./configure --shared --use-cuda=no && \
    make depend -j${NUM_BUILD_CORES} && \
    make -j${NUM_BUILD_CORES}

WORKDIR /

RUN find /opt/kaldi -type f \( -name "*.o" -o name "*.la" -o -name "*.a" \) -exec rm -f {} \; && \
    find /opt/intel -type f -name "*.a" -exec rm -f {} \; && \
    find /opt/intel -type f -regex '.*\(_mc.?\|_mic\|_thread\|_ilp64\)\.so' -exec rm {} \; && \
    rm -rf /kaldi/.git

RUN wget --tries=3 "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh
