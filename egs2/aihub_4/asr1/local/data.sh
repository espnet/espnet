#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=100

#data
datadir=/mnt/ssd/jieun/datasets/
# KsponSpeech
#  |_ KsponSpeech_01/
#  |_ KsponSpeech_02/
#  |_ KsponSpeech_03/
#  |_ KsponSpeech_04/
#  |_ KsponSpeech_05/
#  |_ KsponSpeech_eval/
#  |_ scripts/
# Download data from here:
# https://aihub.or.kr/aidata/105

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: aihub_4 Data Preparation"
    local/trans_prep.sh ${datadir} /mnt/ssd/jieun/datasets
    for x in train dev eval; do
        local/data_prep.sh ${datadir} /mnt/ssd/jieun/datasets data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
