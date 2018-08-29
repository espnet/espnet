FROM espnet/espnet:runtime
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG CUDA_MYR_VERSION=9
ARG CUDA_MNR_VERSION=2
ENV CUDA_PKG_VERSION ${CUDA_MYR_VERSION}-${CUDA_MNR_VERSION}
ENV CUDA_DEV_VERSION ${CUDA_MYR_VERSION}.${CUDA_MNR_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-core-${CUDA_PKG_VERSION} \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-${CUDA_PKG_VERSION} \
        cuda-nvgraph-dev-${CUDA_PKG_VERSION} \
        cuda-cusolver-dev-${CUDA_PKG_VERSION} \
        cuda-cublas-dev-${CUDA_PKG_VERSION} \
        cuda-cufft-dev-${CUDA_PKG_VERSION} \
        cuda-curand-dev-${CUDA_PKG_VERSION} \
        cuda-cusparse-dev-${CUDA_PKG_VERSION} \
        cuda-npp-dev-${CUDA_PKG_VERSION} \
        cuda-driver-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-${CUDA_PKG_VERSION} && \
    ln -s cuda-${CUDA_DEV_VERSION} /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=${CUDA_DEV_VERSION}"

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV CUDNN_VERSION 7.2.1.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7-dev && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# Install NCCL
ENV NCCL_HOME /usr/local/nccl
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make PREFIX=${NCCL_HOME} install && \
    cp ${NCCL_HOME}/include/* ${CUDA_HOME}/include && \
    cp ${NCCL_HOME}/lib/* ${CUDA_HOME}/lib64

# Install espnet gpu
WORKDIR /espnet/tools
RUN make KALDI=/kaldi all 
WORKDIR /