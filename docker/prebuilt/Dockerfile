ARG FROM_IMAGE

# For cuda-based images, The distribution will include cuda, cudnn, nccl
FROM ${FROM_IMAGE}

ARG WITH_PROXY
ARG THIS_USER
ARG THIS_UID
ENV http_proxy ${WITH_PROXY}
ENV https_proxy ${WITH_PROXY}

# Add user to container
RUN if [ ! -z "${THIS_UID}"  ];then \
    useradd -m -r -u ${THIS_UID} -g root ${THIS_USER}; \
    fi

USER ${THIS_USER}
WORKDIR /