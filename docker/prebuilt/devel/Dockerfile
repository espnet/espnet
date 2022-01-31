ARG FROM_TAG

ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

## FROM CUDA 11.1 base 
## [https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.1.1/ubuntu20.04-x86_64/base/Dockerfile] 
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.1.1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    && ln -s cuda-11.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"

ENV CUDA_HOME /usr/local/cuda

## FROM CUDA 11.1 devel
## [https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.1.1/ubuntu20.04-x86_64/devel/Dockerfile]
ENV NCCL_VERSION 2.8.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-1=11.1.74-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-minimal-build-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \
    cuda-nvml-dev-11-1=11.1.74-1 \
    libnpp-dev-11-1=11.1.2.301-1 \
    libcublas-dev-11-1=11.3.0.106-1 \
    libcusparse-dev-11-1=11.3.0.10-1 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold libcublas-dev-11-1
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs


WORKDIR /

ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG CUDA_VER
WORKDIR /

# IF using a local ESPNet repository, a temporary file containing the ESPnet git repo is copied over
ARG ESPNET_ARCHIVE=./espnet-local.tar
COPY  ${ESPNET_ARCHIVE} /espnet-local.tar


# Download ESPnet
RUN echo "Getting ESPnet sources from local repository, in temporary file: " ${ESPNET_ARCHIVE}
RUN mkdir /espnet
RUN tar xf espnet-local.tar -C /espnet/
RUN rm espnet-local.tar

RUN cd espnet && \
    rm -rf docker egs test utils

# Install espnet
WORKDIR /espnet/tools


FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG CUDA_VER
ENV CUDA_VER ${CUDA_VER}

ARG TH_VERSION
ENV TH_VERSION ${TH_VERSION}
WORKDIR /

ARG ESPNET_LOCATION=https://github.com/espnet/espnet

ENV PATH=/opt/miniconda/bin:${PATH}

# Download ESPnet
RUN git clone ${ESPNET_LOCATION} && \
    cd espnet && \
    rm -rf docker egs egs2 espnet2 test utils && \
    rm -rf .git 

# Install espnet
WORKDIR /espnet/tools

# Disable cupy test
# Docker build does not load libcuda.so.1
# Replace nvidia-smi for nvcc because docker does not load nvidia-smi
RUN if [ -z "${CUDA_VER}" ]; then \
        echo "Build without CUDA" && \
        MY_OPTS='CUPY_VERSION=""'; \
    else \
        echo "Build with CUDA ${CUDA_VER}" && \
        # Docker containers cannot load cuda libs during build.
        # So, their checks on cuda packages are disabled.
        sed -i '200s|install.py|install.py --no-cuda --no-cupy |' Makefile && \
        export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}" && \
        MY_OPTS="CUDA_VERSION=${CUDA_VER}" && \
        . ./setup_cuda_env.sh /usr/local/cuda;  \
    fi; \
    if [ ! -z "${TH_VERSION}" ]; then \
        MY_OPTS="${MY_OPTS} TH_VERSION=${TH_VERSION} "; \
    fi; \
    echo "Make with options ${MY_OPTS}" && \
    ln -s /opt/kaldi ./ && \
   rm -f activate_python.sh && touch activate_python.sh && \
   conda install -y conda "python=3.7.4" && \
    make KALDI=/opt/kaldi ${MY_OPTS} && \
    . ./activate_python.sh && \
    ./installers/install_warp-ctc.sh && \
    ./installers/install_kenlm.sh && \
    # ./installers/install_chainer_ctc.sh && \
    conda clean --all && \
    rm -f *.tar.*  && \
    pip cache purge

RUN rm -rf ../espnet

WORKDIR /
