#!/usr/bin/env bash

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
text_format=raw
nj=8
g2p=espeak_ng_russian

log "$0 $*"
# shellcheck disable=SC1091
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;
# shellcheck disable=SC1091
. ./db.sh || exit 1;

if [ -z "${RUSLAN}" ]; then
   log "Fill the value of 'RUSLAN' of db.sh"
   exit 1
fi

db_root=${RUSLAN}
train_set=tr_no_dev
dev_set=dev
eval_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/data_download.sh"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}/RUSLAN" data/all
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    utils/subset_data_dir.sh data/all 500 data/deveval
    utils/subset_data_dir.sh --first data/deveval 250 "data/${dev_set}"
    utils/subset_data_dir.sh --last data/deveval 250 "data/${eval_set}"
    utils/copy_data_dir.sh data/all "data/${train_set}"
    utils/filter_scp.pl --exclude data/deveval/wav.scp \
        data/all/wav.scp > "data/${train_set}/wav.scp"
    utils/fix_data_dir.sh "data/${train_set}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${text_format}" = phn ]; then
    log "stage 2: pyscripts/utils/convert_text_to_phn.py"
    for dset in "${train_set}" "${dev_set}" "${eval_set}"; do
        utils/copy_data_dir.sh "data/${dset}" "data/${dset}_phn"
        pyscripts/utils/convert_text_to_phn.py --g2p "${g2p}" --nj "${nj}" \
            "data/${dset}/text" "data/${dset}_phn/text"
        utils/fix_data_dir.sh "data/${dset}_phn"
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
