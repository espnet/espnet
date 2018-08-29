FROM espnet/espnet:runtime
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

# Install NCCL
ENV NCCL_HOME /usr/local/nccl
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && make PREFIX=${NCCL_HOME} install && \
    cp ${NCCL_HOME}/include/* ${CUDA_HOME}/include && \
    cp ${NCCL_HOME}/lib/* ${CUDA_HOME}/lib64

WORKDIR /espnet/tools
RUN make kaldi USE_VENV=OFF && \
    rm -r ./kaldi/tools/openfst-*/src && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; && \
    find ./kaldi/src -name "*.o" -exec rm -f {} \; && \
    cd kaldi/tools && ./extras/install_beamformit.sh

RUN make nkf USE_VENV=OFF && make kaldi-io-for-python USE_VENV=OFF
RUN make venv/lib/python2.7/site-packages/torch USE_VENV=OFF && make warp-ctc USE_VENV=OFF
RUN make chainer_ctc USE_VENV=OFF && make sentencepiece USE_VENV=OFF

WORKDIR /