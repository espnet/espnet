ARG FROM_TAG=dev3-cpu
FROM espnet/espnet:${FROM_TAG}

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
    fi && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/${USERNAME}/.bashrc && \
    chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
WORKDIR /workspaces
