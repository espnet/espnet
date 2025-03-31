FROM ubuntu:22.04
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
        build-essential \
        automake \
        autotools-dev \
        cmake \
        libffi-dev \
        libtool \
        gnupg2 \
        libncurses5-dev \
        software-properties-common \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        pandoc ffmpeg bc nodejs npm \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Latest version of git
ENV TZ=Etc/UTC
ENV TH_VERSION=2.0.1
ENV USE_CONDA=false
ENV CHAINER_VERSION=6.0.0
ENV PATH=/opt/miniconda/bin:${PATH}

RUN add-apt-repository ppa:git-core/ppa -y && \
    apt update && \
    apt install -y --no-install-recommends git-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --tries=3 -nv "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda config --prepend channels https://software.repos.intel.com/python/conda/ && \
    rm miniconda.sh && \
    conda install -y python=3.10 && \
    conda clean -a -y

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/${USERNAME}/.bashrc

RUN mkdir -p /workspaces && \
    chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
WORKDIR /workspaces
