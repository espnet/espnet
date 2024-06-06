#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${HIFITTS}" ]; then
   log "Fill the value of 'HIFITTS' of db.sh"
   exit 1
fi
db_root=${HIFITTS}

train_set="tr_no_dev"
dev_set="dev"
eval_set="test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.py"

    python local/data_prep.py \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --test_set "${eval_set}" \
        --dataset_path "${db_root}"/hi_fi_tts_v0 \
        --dest_path "data"
fi

utils/fix_data_dir.sh data/${train_set}
utils/fix_data_dir.sh data/${dev_set}
utils/fix_data_dir.sh data/${eval_set}
