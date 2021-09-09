#!/usr/bin/env bash

# Copyright 2021 Takaaki Saeki
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
threshold=35
nj=8
ctcscore_pruning_threshold=-0.1

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${JTUBESPEECH}" ]; then
   log "Fill the value of 'JTUBESPEECH' of db.sh"
   exit 1
fi
db_root=${JTUBESPEECH}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    # Initial normalization of the data
    # Doesn't change sampling frequency and it's done after stages
    local/data_prep.sh "${db_root}/jtuberaw" "${db_root}/jtubesplit" data/all "${ctcscore_pruning_threshold}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    # shellcheck disable=SC2154
    scripts/audio/trim_silence.sh \
        --cmd "${train_cmd}" \
        --nj "${nj}" \
        --fs 16000 \
        --win_length 1024 \
        --shift_length 256 \
        --threshold "${threshold}" \
        data/all data/all/log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh data/all 500 data/devtest
    utils/subset_data_dir.sh --first data/devtest 250 data/dev
    utils/subset_data_dir.sh --last data/devtest 250 data/test
    utils/copy_data_dir.sh data/all data/tr_no_dev
    utils/filter_scp.pl --exclude data/devtest/wav.scp \
        data/all/wav.scp > data/tr_no_dev/wav.scp
    utils/fix_data_dir.sh data/tr_no_dev
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
