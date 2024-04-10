#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100000

tgt_lang=$1  # one of hi (Hindi), bn (Bengali), or ta (Tamil)
remove_archive=false

log "$0 $*"
. utils/parse_options.sh

if [ -z "${IWSLT24_INDIC}" ]; then
    log "Please fill the value of 'IWSLT24_INDIC' of db.sh to indicate where the dataset zip files are downloaded."
    exit 1
fi

if [ $# -ne 1 ]; then
    log "Usage: $0 <tgt_lang>"
    log "e.g.: $0 hi"
    exit 1
fi

# check tgt_lang
if [ "$tgt_lang" == "hi" ]; then
    target_language="Hindi"
elif [ "$tgt_lang" == "bn" ]; then
    target_language="Bengali"
elif [ "$tgt_lang" == "ta" ]; then
    target_language="Tamil"
else
    log "Error: ${tgt_lang} is not supported. It must be one of hi, bn, or ta."
    exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1.1: Data download and unpack"
    mkdir -p ${IWSLT24_INDIC}
    local/download_and_unpack.sh ${IWSLT24_INDIC} ${tgt_lang} ${remove_archive}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 1.2: ESPnet data format preparation"
    # TODO: local/data_prep.sh ${IWSLT24_INDIC} ${tgt_lang}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
