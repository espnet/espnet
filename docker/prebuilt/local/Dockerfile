ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

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

# Disable cupy test
# Docker build does not load libcuda.so.1
# Replace nvidia-smi for nvcc because docker does not load nvidia-smi
RUN if [ -z "$( nvcc -V )" ]; then \
        make KALDI=/kaldi CUPY_VERSION=''; \
    else \
        sed -i '159s|install.py|install.py --no-cupy|' Makefile && \
        sed -i '19s|nvidia-smi|nvcc|' Makefile && \
        make KALDI=/kaldi CUDA_VERSION=${CUDA_VER}; \
fi

RUN rm -rf ../espnet

WORKDIR /
