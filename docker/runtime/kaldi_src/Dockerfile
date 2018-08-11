FROM fhrozen/dev_espnet:runtime-kaldi-tools
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

WORKDIR /kaldi

RUN cd src && \ 
    ./configure --shared --use-cuda=no && \
    make depend && \
    make all && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; 

WORKDIR /