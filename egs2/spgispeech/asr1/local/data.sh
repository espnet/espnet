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
stop_stage=100000

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SPGISPEECH}" ]; then
    log "Fill the value of 'SPGISPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: Data Preparation"
    # we assume the following data structure
    # spgispeech
    # ├── train
    # ├── train.csv
    # ├── val
    # └── val.csv
    for part in train val; do
        local/data_prep.sh ${part} ${SPGISPEECH} data/${part}
    done

    utils/fix_data_dir.sh data/train
    utils/fix_data_dir.sh data/val

    utils/fix_data_dir.sh data/train_unnorm
    utils/fix_data_dir.sh data/val_unnorm
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # normalized casea
    # make a development set with the first 4000 utterances
    utils/subset_data_dir.sh --first data/train 4000 data/dev_4k
    n=$(($(wc -l < data/train/text) - 4000))
    utils/subset_data_dir.sh --last data/train ${n} data/train_nodev
    # unnormalized casea
    # make a development set with the first 4000 utterances
    utils/subset_data_dir.sh --first data/train_unnorm 4000 data/dev_4k_unnorm
    n=$(($(wc -l < data/train_unnorm/text) - 4000))
    utils/subset_data_dir.sh --last data/train_unnorm ${n} data/train_nodev_unnorm
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
