#!/bin/bash
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
stop_stage=10

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

data_type=legacy

train_set="train"
train_dev="dev"
test_set="test"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    local/download_data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    local/prepare_data.sh ${data_type}
    for dset in ${train_dev} ${test_set} ${train_set}; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: LM data preparation"
    mkdir -p data/local/lm_train
    gunzip -c db/TEDLIUM_release-3/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py | awk '{print NR, $0}' > data/local/lm_train/train.txt
fi
