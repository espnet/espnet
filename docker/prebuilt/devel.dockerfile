ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG} as devel
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
    ./installers/install_chainer.sh cpu && \
    conda clean --all && \
    rm -f *.tar.*  && \
    pip cache purge

RUN rm -rf ../espnet

WORKDIR /


#### For local docker
FROM devel as espnet_local
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
