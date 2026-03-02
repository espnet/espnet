ARG USE_IMAGE=ubuntu:latest

FROM ${USE_IMAGE}
LABEL maintainer="Nelson Yalta <nyalta21@gmail.com>"

ENV NUM_BUILD_CORES=12
ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN if [ -z "$(getent group ${GROUP_ID})" ]; then \
    groupadd -g ${GROUP_ID} "${USERNAME}"; \
    else \
    existing_group="$(getent group $GROUP_ID | cut -d: -f1)"; \
    if [ "${existing_group}" != "${USERNAME}" ]; then \
    groupmod -n "${USERNAME}" "${existing_group}"; \
    fi; \
    fi && \
    if [ -z "$(getent passwd $USER_ID)" ]; then \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} "${USERNAME}"; \
    else \
    existing_user="$(getent passwd ${USER_ID} | cut -d: -f1)"; \
    if [ "${existing_user}" != "${USERNAME}" ]; then \
    usermod -l "${USERNAME}" -d /home/"${USERNAME}" -m "${existing_user}"; \
    fi; \
    fi

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        automake \
        autotools-dev \
        bc \
        build-essential \
        cmake \
        gawk \
        curl \
        gfortran \
        libffi-dev \
        libtool \
        git \
        gnupg2 \
        libncurses5-dev \
        software-properties-common \
        sox \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        pandoc ffmpeg nodejs npm \
        && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install --no-install-recommends \
        git-lfs \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Latest version of git
ENV TZ=Etc/UTC
ENV TH_VERSION=2.7.1
ENV ESPNET_PYTHON_VERSION=3.12
ENV USE_CONDA=true
ENV CHAINER_VERSION=6.0.0
ENV PATH=/opt/miniconda/bin:${PATH}

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/${USERNAME}/.bashrc

RUN chown -R ${USERNAME}:${USERNAME} /opt
RUN mkdir -p /workspaces && \
    chown -R ${USERNAME}:${USERNAME} /workspaces

WORKDIR /opt
USER ${USERNAME}

WORKDIR /workspaces
