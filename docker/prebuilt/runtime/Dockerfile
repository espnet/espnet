FROM ubuntu:16.04
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ENV DOCKER_BUILT_VER 18.09.1-ce

ARG NUM_BUILD_CORES=8
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install --no-install-recommends \ 
        automake \
        autoconf \
        apt-utils \
        bc \
        build-essential \ 
        cmake \
        curl \
        flac \  
        gawk \ 
        git \
        libasound2-dev \
        libatlas3-base \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-dev \
        libperl-dev \ 
        libpng12-dev \
        libprotobuf-dev \
        libprotobuf9v5 \
        libsndfile1 \
        libsndfile-dev \
        libtool \
        libzmq3-dev \ 
        pciutils \
        perl \
        pkg-config \
        protobuf-compiler \
        python-dev \
        python-tk \
        python-numpy-dev \
        python3-numpy-dev \ 
        software-properties-common \
        sox \
        subversion \
        swig \
        unzip \
        wget \ 
        zip \
        zlib1g-dev \
        && \
    apt-get -y install --reinstall \
        libc6-dev \
        linux-libc-dev \
        libgcc-5-dev \
        libstdc++-5-dev \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get -y install --no-install-recommends \
        python3.6 \
        python3.6-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install pip
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip install -U virtualenv

# Install Kaldi
RUN git clone https://github.com/kaldi-asr/kaldi

RUN cd /kaldi/tools && \
    ./extras/install_mkl.sh -sp debian intel-mkl-64bit-2019.2-057 && \
    make all && \
    rm -r openfst-*/src && \
    ./extras/install_beamformit.sh && \
    ./extras/install_mkl.sh && \
    ./extras/install_irstlm.sh && \
    cd /kaldi/src && \
    ./configure --shared --use-cuda=no && \
    make depend -j${NUM_BUILD_CORES} && \
    make -j${NUM_BUILD_CORES} && \
    find /kaldi/src -name "*.o" -exec rm -f {} \; && \
    find /kaldi/src -name "*.o" -exec rm -f {} \; 

WORKDIR /