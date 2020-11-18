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

# Replace nvidia-smi for nvcc because docker does not load nvidia-smi
RUN if [ -z "$( which nvcc )" ]; then \
        echo "Build without CUDA" && \
        MY_OPTS='CUPY_VERSION=""  TH_VERSION=1.6.0'; \
    else \
        echo "Build with CUDA" && \
        # Disable cupy test
        # Docker build does not load libcuda.so.1
        # So, their checks on cuda packages are disabled.
        sed -i '200s|install.py|install.py --no-cuda --no-cupy |' Makefile && \
        export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}" && \
        MY_OPTS="CUDA_VERSION=${CUDA_VER}" && \
        . ./setup_cuda_env.sh /usr/local/cuda;  \
    fi; \ 
    if [ "${CUDA_VER}" = "10.1" ]; then \
        # warpctc is not supported from Pytorch 1.3.1
        MY_OPTS="${MY_OPTS} TH_VERSION=1.6.0";  \
    fi; \
    echo "Make with options ${MY_OPTS}" && \
    ln -s /kaldi ./ && \
    ./setup_anaconda.sh /miniconda espnet 3.7.4 && \
    make KALDI=/kaldi ${MY_OPTS}

RUN rm -rf ../espnet

WORKDIR /
