ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG CUDA_VER
WORKDIR /

ARG ESPNET_LOCATION=https://github.com/espnet/espnet

# Download ESPnet
RUN git clone ${ESPNET_LOCATION} && \
    cd espnet && \
    rm -rf docker egs test utils

# Install espnet
WORKDIR /espnet/tools

# Disable cupy test
# Docker build does not load libcuda.so.1
# Replace nvidia-smi for nvcc because docker does not load nvidia-smi
RUN if [ -z "$( nvcc -V )" ]; then \
        echo "Build withoug CUDA" &&\
        make KALDI=/kaldi CUPY_VERSION=''; \
    else \
        echo "Build with CUDA" && \
        export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
        sed -i '154s|install.py|install.py --no-cupy|' Makefile && \
        sed -i '21s|nvidia-smi|nvcc|' Makefile && \
        make KALDI=/kaldi CUDA_VERSION=${CUDA_VER}; \
    fi

RUN rm -rf ../espnet

WORKDIR /
