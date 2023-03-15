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
stop_stage=100

#data
datadir=/ocean/projects/cis210027p/shared/corpora/KsponSpeech/KsponSpeech/
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
    log "stage 1: KsponSpeech Data Preparation"
    local/trans_prep.sh ${datadir} data/local/KsponSpeech
    for x in train dev eval_clean eval_other; do
        local/data_prep.sh ${datadir} data/local/KsponSpeech data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
