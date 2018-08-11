FROM fhrozen/dev_espnet:runtime-kaldi-tools
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG NUM_BUILD_CORES=8
ENV NUM_BUILD_CORES ${NUM_BUILD_CORES}

RUN pushd /kaldi/src; ./configure --shared --use-cuda=no && \
    make depend && \
    make -j${NUM_BUILD_CORES}; popd

RUN find /kaldi/src -name "*.o" -exec rm -f {} \; && \
    find /kaldi/src -name "*.o" -exec rm -f {} \; 

WORKDIR /