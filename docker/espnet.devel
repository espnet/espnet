ARG FROM_IMAGE

#For cuda-based images, The distribution will include cuda, cudnn, nccl
FROM ${FROM_IMAGE}

ARG WITH_PROXY
ARG THIS_USER
ARG THIS_UID
ENV http_proxy ${WITH_PROXY}
ENV https_proxy ${WITH_PROXY}
ENV CUDA_HOME /usr/local/cuda
ENV CUDA_PATH ${CUDA_HOME}
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
        software-properties-common \
        sox \
        subversion \
        wget \ 
        swig \
        zip \
        zlib1g-dev \
        && \
    apt-get -y install --reinstall \
        libc6-dev \
        linux-libc-dev \
        libgcc-5-dev \
        libstdc++-5-dev \
        && \
    apt-get -y remove \
        libnccl-dev \
        libnccl2 \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get -y install --no-install-recommends \
        python3.6 \
        python3.6-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install NCCL
ENV NCCL_HOME /usr/local/nccl
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make PREFIX=${NCCL_HOME} install && \
    cp ${NCCL_HOME}/include/* ${CUDA_HOME}/include && \
    cp ${NCCL_HOME}/lib/* ${CUDA_HOME}/lib64

# Add user to container
RUN if [ ! -z "${THIS_UID}"  ];then \
    useradd -m -r -u ${THIS_UID} -g root ${THIS_USER}; \
    fi

# Install pip
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
RUN pip install -U soundfile virtualenv

# Install espnet
RUN mkdir -p /espnet
COPY ./tools /espnet/tools

WORKDIR /espnet/tools
RUN make kaldi USE_VENV=OFF && \
    rm -r ./kaldi/tools/openfst-*/src && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; && \
    cd kaldi/tools && ./extras/install_beamformit.sh

RUN make nkf USE_VENV=OFF && make kaldi-io-for-python USE_VENV=OFF
RUN make venv/lib/python2.7/site-packages/torch USE_VENV=OFF && make warp-ctc USE_VENV=OFF
RUN make chainer_ctc USE_VENV=OFF && make sentencepiece USE_VENV=OFF

USER ${THIS_USER}
WORKDIR /