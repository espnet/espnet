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

<<<<<<< HEAD
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
=======
# Delete cupy if cpu image
RUN if [ -z "$( nvcc -V )" ];then \
        sed -i 's| cupy==$(CUPY_VERSION) torch|torch|' Makefile && \
        sed -i '\|. venv/bin/activate && pip install cupy==$(CUPY_VERSION)|d' Makefile; \
    fi 

# Delete cupy test
# Docker build does not load libcuda.so.1
RUN sed -i "/cupy'/d" check_install.py; 

RUN make KALDI=/kaldi && \
    rm -rf ../espnet
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986

WORKDIR /
