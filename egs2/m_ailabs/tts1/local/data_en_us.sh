#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

stage=-1
stop_stage=1

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

if [ -z "${M_AILABS}" ]; then
    log "Fill the value of 'JSUT' of db.sh"
    exit 1
fi
db_root=${M_AILABS}

# silence part trimming related
do_trimming=true
nj=32

# dataset configuration
lang="en_US"
spkr="judy"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Downloading data"
    local/download.sh ${db_root} ${lang}
fi

org_set=${lang}_${spkr}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"
    local/data_prep.sh ${db_root} ${lang} ${spkr} data/${org_set}
    utils/fix_data_dir.sh data/${org_set}
    utils/validate_data_dir.sh --no-feats data/${org_set}

    # Trim silence parts at the beginning and the end of audio
    if ${do_trimming}; then
        log "Trimmng silence."
        scripts/audio/trim_silence.sh \
            --cmd "${train_cmd}" \
            --nj "${nj}" \
            --fs 16000 \
            --win_length 1024 \
            --shift_length 256 \
            --threshold 60 \
            --min_silence 0.01 \
            data/${org_set} \
            data/${org_set}/log
    fi

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dividing into subsets"

    train_set="tr_no_dev"
    dev_set="dev"
    eval_set="eval"
    utils/subset_data_dir.sh --last data/${org_set} 50 data/${org_set}_tmp
    utils/subset_data_dir.sh --last data/${org_set}_tmp 25 data/${dev_set}
    utils/subset_data_dir.sh --first data/${org_set}_tmp 25 data/${eval_set}
    n=$(($(wc -l <data/${org_set}/wav.scp) - 50))
    utils/subset_data_dir.sh --first data/${org_set} ${n} data/${train_set}
    rm -rf data/${org_set}_tmp
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
