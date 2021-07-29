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
threshold=35
nj=32
g2p=g2pk_no_space

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

if [ -z "${KSS}" ]; then
   log "Fill the value of 'KSS' of db.sh"
   exit 1
fi

db_root=${KSS}
train_set=tr_no_dev
dev_set=dev
eval_set=eval1

if [ ! -e "${KSS}/transcript.v.1.4.txt" ]; then
    log "KSS dataset is not found."
    log "Please download it from https://bit.ly/376oCzY and locate as follows:"
    cat << EOF
$ vim db.sh
KSS=/path/to/kss

$ tree -L 1 /path/to/kss
/path/to/kss
├── 1
├── 2
├── 3
├── 4
└── transcript.v.1.4.txt
EOF
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}" data/all
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    # shellcheck disable=SC2154
    scripts/audio/trim_silence.sh \
        --cmd "${train_cmd}" \
        --nj "${nj}" \
        --fs 44100 \
        --win_length 2048 \
        --shift_length 512 \
        --threshold "${threshold}" \
        data/all data/all/log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    utils/subset_data_dir.sh data/all 500 data/deveval
    utils/subset_data_dir.sh --first data/deveval 250 "data/${dev_set}"
    utils/subset_data_dir.sh --last data/deveval 250 "data/${eval_set}"
    utils/copy_data_dir.sh data/all "data/${train_set}"
    utils/filter_scp.pl --exclude data/deveval/wav.scp \
        data/all/wav.scp > "data/${train_set}/wav.scp"
    utils/fix_data_dir.sh "data/${train_set}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ "${text_format}" = phn ]; then
    log "stage 3: pyscripts/utils/convert_text_to_phn.py"
    for dset in "${train_set}" "${dev_set}" "${eval_set}"; do
        utils/copy_data_dir.sh "data/${dset}" "data/${dset}_phn"
        pyscripts/utils/convert_text_to_phn.py --g2p "${g2p}" --nj "${nj}" \
            "data/${dset}/text" "data/${dset}_phn/text"
        utils/fix_data_dir.sh "data/${dset}_phn"
    done
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
