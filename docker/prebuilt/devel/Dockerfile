ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

WORKDIR /

# Download ESPnet
RUN git clone https://github.com/espnet/espnet && \
    cd espnet && \
    rm -rf docker egs test utils

# Install espnet
WORKDIR /espnet/tools

# Delete cupy if cpu image
RUN if [ -z "$( nvcc -V )" ];then \
        sed -i 's| cupy==$(CUPY_VERSION) torch|torch|' Makefile && \
        sed -i '\|. venv/bin/activate && pip install cupy==$(CUPY_VERSION)|d' Makefile; \
    fi 

RUN make KALDI=/kaldi && \
    rm -rf ../espnet

WORKDIR /