ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

WORKDIR /

ARG ESPNET_LOCATION=https://github.com/espnet/espnet

# IF using a local ESPNet repository, a temporary file containing the ESPnet git repo is copied over
# As temporary build mounts are not easily possible within docker,
# using a conditional copy is a workaround https://stackoverflow.com/a/46801962
ARG ESPNET_ARCHIVE=./espnet-local.ta[r]
COPY  ./run.sh ${ESPNET_ARCHIVE} /
RUN rm /run.sh


# Download ESPnet
RUN if test ${ESPNET_LOCATION}="local" ; then \
        echo "Getting ESPnet sources from local repository." \
        test -r ${ESPNET_ARCHIVE} && echo "Using Archive " ${ESPNET_ARCHIVE} ; \
        mkdir /espnet && tar -xf ${ESPNET_ARCHIVE} -C /espnet/ ; \
    else \
        echo "Cloning ESPnet from " ${ESPNET_LOCATION} \
        git clone ${ESPNET_LOCATION} ; \
    fi

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
        sed -i '158s|install.py|install.py --no-cupy|' Makefile && \
        sed -i '19s|nvidia-smi|nvcc|' Makefile && \
        make KALDI=/kaldi; \
    fi

RUN rm -rf ../espnet

WORKDIR /
