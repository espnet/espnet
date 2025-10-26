ARG FROM_TAG=gpu-latest
ARG NUM_BUILD_CORES=8
ARG DOCKER_VER
FROM espnet/espnet:${FROM_TAG} AS cuda_builder
LABEL maintainer="Nelson Yalta <nyalta21@gmail.com>"

## FROM CUDA 12.6.3 base
## [https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubuntu2404/base/Dockerfile]
ENV NVARCH=x86_64

ENV NVIDIA_REQUIRE_CUDA="cuda>=12.6 brand=unknown,driver>=470,driver<471 brand=grid,driver>=470,driver<471 brand=tesla,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=vapps,driver>=470,driver<471 brand=vpc,driver>=470,driver<471 brand=vcs,driver>=470,driver<471 brand=vws,driver>=470,driver<471 brand=cloudgaming,driver>=470,driver<471 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551"
ENV NV_CUDA_CUDART_VERSION=12.6.77-1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=12.6.3

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-6=${NV_CUDA_CUDART_VERSION} \
    cuda-compat-12-6 \
    && rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV CUDA_HOME=/usr/local/cuda

## FROM CUDA 12.6.3 devel
## [https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubuntu2404/devel/Dockerfile]
ENV NV_CUDA_LIB_VERSION="12.6.3-1"
ENV NV_CUDA_CUDART_DEV_VERSION=12.6.77-1
ENV NV_NVML_DEV_VERSION=12.6.77-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.4.2-1
ENV NV_LIBNPP_DEV_VERSION=12.3.1.54-1
ENV NV_LIBNPP_DEV_PACKAGE="libnpp-dev-12-6=${NV_LIBNPP_DEV_VERSION}"

ENV NV_NVPROF_VERSION=12.6.80-1
ENV NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-6=${NV_NVPROF_VERSION}"

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-6
ENV NV_LIBCUBLAS_DEV_VERSION=12.6.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE="${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}"

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.23.4-1
ENV NCCL_VERSION=2.23.4-1
ENV NV_LIBNCCL_DEV_PACKAGE="${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.6"

# ## FROM CUDA 12.6.3 runtime
# [https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.3/ubuntu2004/runtime/Dockerfile]
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.23.4-1
ENV NCCL_VERSION=2.23.4-1
ENV NV_LIBNCCL_PACKAGE="${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.6"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-6=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-6=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-6=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

WORKDIR /
