#!/usr/bin/env bash
# Copyright 2023 ISSAI (author: Yerbolat Khassanov)
# Apache 2.0

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

log "$0 $*"
. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ -z "${KSC}" ]; then
  log "Error: \$KSC is not set in db.sh."
  exit 2
fi

log "Download data to ${KSC}"
if [ ! -d "${KSC}" ]; then
    mkdir -p "${KSC}"
fi

#absolute path
KSC=$(cd ${KSC}; pwd)

echo local/download_data.sh
local/download_data.sh "${KSC}"

echo local/prepare_data.sh
local/prepare_data.sh "${KSC}"

log "Successfully finished. [elapsed=${SECONDS}s]"
