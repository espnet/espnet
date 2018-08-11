FROM fhrozen/dev_espnet:runtime-utils
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

WORKDIR /kaldi

RUN cd tools && \
    make all && \
    rm -r openfst-*/src && \
    ./extras/install_beamformit.sh

WORKDIR /