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
SECONDS=0

stage=1
stop_stage=100000

recog_set="US UK IND CHN JPN PT RU KR CA ES"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${AESRC2020}" ]; then
    log "Fill the value of 'AESRC2020' of db.sh"    # place datatang.zip in this dir
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -d "${AESRC2020}/Datatang-English/data" ]; then
        echo "Data Download to ${AESRC2020}"
        if [ ! -f "${AESRC2020}/datatang.zip" ]; then
            echo "The AESRC2020 data needs to be requested via services@datatang.com"
            exit 1
        fi
        local/download_and_untar.sh ${AESRC2020}/datatang.zip ${AESRC2020}
    else
        log "stage 1: ${AESRC2020}/Datatang-English/data is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Data preparation"
    local/data_prep.sh ${AESRC2020}/Datatang-English/data data
    ./utils/fix_data_dir.sh data/data_all
    local/create_subsets.sh data "${recog_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
