ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG CUDA_VER
WORKDIR /

ARG ESPNET_LOCATION=https://github.com/espnet/espnet

# Download ESPnet
RUN git clone ${ESPNET_LOCATION} && \
    cd espnet && \
    rm -rf docker egs egs2 espnet2 test utils

# Install espnet
WORKDIR /espnet/tools

# Disable cupy test
# Docker build does not load libcuda.so.1
# Replace nvidia-smi for nvcc because docker does not load nvidia-smi
RUN if [ -z "$( nvcc -V )" ]; then \
        echo "Build without CUDA" && \
        MY_OPTS="CUPY_VERSION="; \
    else \
        echo "Build with CUDA" && \
        export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}" && \
        MY_OPTS="CUDA_VERSION=${CUDA_VER}" && \
        sed -i 's|_install.py|_install.py --no-cuda|g' Makefile && \
        sed -i 's|which nvidia-smi|which nvcc|g' Makefile;  \
    fi; \ 
    if [ "${CUDA_VER}" = "10.1" ]; then \
        # Pytorch 1.3.1 is not supported by warpctc
        MY_OPTS="${MY_OPTS} TH_VERSION=1.3.1";  \
    fi; \
    echo "Make with options ${MY_OPTS}"; \
    make KALDI=/kaldi ${MY_OPTS}

RUN rm -rf ../espnet

WORKDIR /
