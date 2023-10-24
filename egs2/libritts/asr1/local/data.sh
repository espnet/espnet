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

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${LIBRITTS}" ]; then
   log "Fill the value of 'LIBRITTS' of db.sh"
   exit 1
fi
db_root=${LIBRITTS}
data_url=www.openslr.org/resources/60

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: local/donwload_and_untar.sh"
    # download the original corpus
    if [ ! -e "${db_root}"/LibriTTS/.complete ]; then
        for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
            local/download_and_untar.sh "${db_root}" "${data_url}" "${part}"
        done
        touch "${db_root}/LibriTTS/.complete"
    else
        log "Already done. Skiped."
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    for name in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
        # Create kaldi data directory with the original audio
        local/data_prep.sh "${db_root}/LibriTTS/${name}" "data/${name}"
        utils/fix_data_dir.sh "data/${name}"
        # Convert graphemes to phonemes
        local/phonemize_dir.py "data/${name}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/combine_data.sh"
    utils/combine_data.sh data/dev data/dev-clean data/dev-other
    utils/combine_data.sh data/train-960 data/train-clean-100 data/train-clean-360 data/train-other-500
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
