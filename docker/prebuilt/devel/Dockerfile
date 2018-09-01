ARG FROM_TAG
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

COPY requirements.txt /
# Install espnet
WORKDIR /

RUN if [ -z "$( nvcc -V )" ];then \
        rm /espnet/tools/requirements.txt && \
        cp requirements.txt /espnet/tools; \
    else \
        rm requirements.txt; \
    fi 

WORKDIR /espnet/tools
RUN make KALDI=/kaldi all 

WORKDIR /