ARG FROM_TAG
ARG NUM_BUILD_CORES=8
ARG DOCKER_VER
FROM espnet/espnet:${FROM_TAG} AS cuda_builder
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
