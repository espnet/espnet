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

spk=$1

available_spks=(
    "slt" "clb" "bdl" "rms" "jmk" "awb" "ksp" "all"
)

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <spk>"
    echo "Available speakers: ${available_spks[*]}"
    exit 2
fi

# check speakers
# shellcheck disable=SC2048
if ! eval "$(echo ${available_spks[*]} | grep -q ${spk})"; then
    echo "Specified spk (${spk}) is not available or not supported." >&2
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

db_root=${CMU_ARCTIC}

train_set=${spk}_train_no_dev
dev_set=${spk}_dev
eval_set=${spk}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    if [ ${spk} = "all" ] ; then
        local/data_download_all.sh "${db_root}"
    else
        local/data_download.sh "${db_root}" "${spk}"
    fi
    
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    if [ ${spk} = "all" ] ; then
        local/data_prep_all.sh \
            --train_set "${train_set}" \
            --dev_set "${dev_set}" \
            --eval_set "${eval_set}" \
            "${db_root}/cmu_us_${spk}_arctic"
    else
        local/data_prep.sh \
            --train_set "${train_set}" \
            --dev_set "${dev_set}" \
            --eval_set "${eval_set}" \
            "${db_root}/cmu_us_${spk}_arctic" "${spk}"
    fi
fi
