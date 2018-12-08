ARG FROM_TAG
# For cuda-based images, The distribution will include cuda, cudnn, nccl
FROM espnet/espnet:${FROM_TAG}
LABEL maintainer "Nelson Yalta <nyalta21@gmail.com>"

ARG THIS_USER
ARG THIS_UID

# Add user to container
RUN if [ ! -z "${THIS_UID}"  ];then \
    useradd -m -r -u ${THIS_UID} -g root ${THIS_USER}; \
    fi

USER ${THIS_USER}
WORKDIR /