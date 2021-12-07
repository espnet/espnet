#!/usr/bin/env bash
# Copyright 2021 UCAS (Author: Keqi Deng)
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

if [ -z "${TEDLIUM2}" ]; then
  log "Error: \$TEDLIUM2 is not set in db.sh."
  exit 2
fi

log "Download data to ${TEDLIUM2}"
if [ ! -d "${TEDLIUM2}" ]; then
    mkdir -p "${TEDLIUM2}"
fi

#To absolute path
TEDLIUM2=$(cd ${TEDLIUM2}; pwd)

echo local/download_data.sh
local/download_data.sh "${TEDLIUM2}"
echo local/prepare_data.sh
local/prepare_data.sh "${TEDLIUM2}"
for dset in dev test train; do
utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
done

mkdir -p data/train data/dev data/test

log "Successfully finished. [elapsed=${SECONDS}s]"
