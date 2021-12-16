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
spk=jvs010

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${JVS}" ]; then
   log "Fill the value of 'JVS' of db.sh"
   exit 1
fi
db_root=${JVS}

train_set=${spk}_tr_no_dev
train_dev=${spk}_dev
eval_set=${spk}_eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/data_download.sh"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh "${db_root}"/jvs_ver1 "${spk}" "data/${spk}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/copy_data_dir.sh "data/${spk}_parallel100" "data/${train_set}"
    utils/subset_data_dir.sh --first "data/${spk}_nonpara30" 15 "data/${train_dev}"
    utils/subset_data_dir.sh --last "data/${spk}_nonpara30" 15 "data/${eval_set}"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
