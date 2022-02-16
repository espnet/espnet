#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
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

# Data preparation related
data_url=www.openslr.org/resources/62
remove_archive=false
download_opt=

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${AIDATATANG_200ZH}" ]; then
  log "Error: \$AIDATATANG_200ZH is not set in db.sh."
  exit 2
fi


log "Download data to ${AIDATATANG_200ZH}"
if [ ! -d "${AIDATATANG_200ZH}" ]; then
    mkdir -p "${AIDATATANG_200ZH}"
fi
# To absolute path
AIDATATANG_200ZH=$(cd ${AIDATATANG_200ZH}; pwd)

local/download_and_untar.sh ${download_opt} "${AIDATATANG_200ZH}" "${data_url}" aidatatang_200zh

log "Data Preparation"
local/data_prep.sh ${AIDATATANG_200ZH}/aidatatang_200zh/corpus ${AIDATATANG_200ZH}/aidatatang_200zh/transcript

for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
    rm data/${x}/text.org
done

log "Successfully finished. [elapsed=${SECONDS}s]"
