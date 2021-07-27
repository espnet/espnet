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
spk=Hokuspokus

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

if [ -z "${HUI_ACG}" ]; then
   log "Fill the value of 'HUI_ACG' of db.sh"
   exit 1
fi

db_root=${HUI_ACG}
train_set=${spk,,}_tr_no_dev
dev_set=${spk,,}_dev
eval_set=${spk,,}_eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/data_download.sh"
    local/data_download.sh "${db_root}" "${spk}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}" "${spk}" "data/${spk,,}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/subset_data_dir.sh"
    utils/subset_data_dir.sh "data/${spk,,}" 500 "data/${spk,,}_deveval"
    utils/subset_data_dir.sh --first "data/${spk,,}_deveval" 250 "data/${dev_set}"
    utils/subset_data_dir.sh --last "data/${spk,,}_deveval" 250 "data/${eval_set}"
    utils/copy_data_dir.sh "data/${spk,,}" "data/${train_set}"
    utils/filter_scp.pl --exclude "data/${spk,,}_deveval/wav.scp" \
        "data/${spk,,}/wav.scp" > "data/${train_set}/wav.scp"
    utils/fix_data_dir.sh "data/${train_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
