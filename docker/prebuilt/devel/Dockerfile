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
        echo "Build withoug CUDA" && \
        MY_OPTS="CUPY_VERSION=''"; \
    else \
        echo "Build with CUDA" && \
        export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}" && \
        MY_OPTS="CUDA_VERSION=${CUDA_VER}" && \
        sed -i '154s|install.py|install.py --no-cupy|' Makefile && \
        sed -i '21s|nvidia-smi|nvcc|' Makefile;  \
    fi; \ 
    if [ "${CUDA_VER}" = "10.1" ]; then \
        sed -i '23s|1.0.1.post2|1.3.1|' check_install.py && \
        # Pytorch 1.3.1 is not supported by warpctc 
        # so warpctc is remove from the check
        sed -i '26d' check_install.py && \
        MY_OPTS="${MY_OPTS} TH_VERSION=1.3.1";  \
    fi; \ 
    echo "Make with options ${MY_OPTS}" && make KALDI=/kaldi ${MY_OPTS}

RUN rm -rf ../espnet

WORKDIR /
