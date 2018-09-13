ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

WORKDIR /

# Download ESPnet
RUN git clone https://github.com/espnet/espnet && \
    cd espnet && \
    rm -rf docker egs src test

# Install espnet
WORKDIR /espnet/tools

# Delete cupy if cpu image
RUN if [ -z "$( nvcc -V )" ];then \
        sed -i '/cupy*/d' requirements.txt; \
    fi 

RUN make KALDI=/kaldi all 

WORKDIR /