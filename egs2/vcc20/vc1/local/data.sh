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
stop_stage=1

log "$0 $*"
. utils/parse_options.sh

srcspk=$1
trgspk=$2

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <srcspk> <trgspk>"
    exit 2
fi

# check speakers
# shellcheck disable=SC2048
if ! eval "$(echo ${available_spks[*]} | grep -q ${srcspk})"; then
    echo "Specified srcspk (${srcspk}) is not available or not supported." >&2
    exit 2
fi
# shellcheck disable=SC2048
if ! eval "$(echo ${available_spks[*]} | grep -q ${trgspk})"; then
    echo "Specified trgspk (${trgspk}) is not available or not supported." >&2
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${VCC20}" ]; then
   log "Fill in the value of 'VCC20' in db.sh"
   exit 1
fi
db_root=${VCC20}

src_train_set=${srcspk}_train
src_dev_set=${srcspk}_dev
src_eval_set=${srcspk}_eval
trg_train_set=${trgspk}_train
trg_dev_set=${trgspk}_dev
trg_eval_set=${trgspk}_eval

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    local/data_prep.sh \
        --train_set "${src_train_set}" \
        --dev_set "${src_dev_set}" \
        --eval_set "${src_eval_set}" \
        "${db_root}/cmu_us_${srcspk}_arctic" "${trgspk}"
fi