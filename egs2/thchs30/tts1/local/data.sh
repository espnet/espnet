#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=2
threshold=35
nj=16

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${THCHS30}" ]; then
   log "Fill the value of 'THCHS30' of db.sh"
   exit 1
fi
db_root=${THCHS30}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage -1: download data from openslr"
    local/download_and_untar.sh "${db_root}" "https://www.openslr.org/resources/18/data_thchs30.tgz" data_thchs30.tgz
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: prepare thchs30 data"
    local/thchs-30_data_prep.sh "$(pwd)" "${db_root}"/data_thchs30 || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: scripts/audio/trim_silence.sh"
    for x in train dev test train_phn dev_phn test_phn; do
        # shellcheck disable=SC2154
        scripts/audio/trim_silence.sh \
             --cmd "${train_cmd}" \
             --nj "${nj}" \
             --fs 16000 \
             --win_length 512 \
             --shift_length 128 \
             --threshold "${threshold}" \
             data/${x} data/${x}/log

        utils/fix_data_dir.sh data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
