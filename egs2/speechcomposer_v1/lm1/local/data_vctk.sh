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

if [ $# -ne 1 ]; then
    log "Error: data_dir required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${VCTK}" ]; then
   log "Fill the value of 'VCTK' of db.sh"
   exit 1
fi
db_root=${VCTK}

train_set=train
dev_set=
eval_set=

vctk_dir="local/vctk"
data_dir=$1
data_dir="${data_dir}/tts"
mkdir -p ${data_dir}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    ${vctk_dir}/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    ${vctk_dir}/data_prep.sh \
        --train_set "${train_set}" --dev_set "${dev_set}" --eval_set "${eval_set}" \
        --num_dev 0 --num_eval 0 \
        "${db_root}"/VCTK-Corpus "${data_dir}"

fi
