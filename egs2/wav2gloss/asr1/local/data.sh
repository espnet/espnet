#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=100
langs=  # comma separated list of languages
tasks=  # comma separated list of tasks
SECONDS=0


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./utils/parse_options.sh

log "data preparation started"

if [ -z "${WAV2GLOSS}" ]; then
    log "Fill the value of 'WAV2GLOSS' of db.sh"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Download Data to ${WAV2GLOSS}"

    git lfs clone https://huggingface.co/datasets/taiqihe/cocoon-gloss ${WAV2GLOSS}

    # untar everything
    for f in ${WAV2GLOSS}/data/*/audio; do
        for split in train dev test; do
            tar -xvf "${f}/${split}.tar.gz" -C "${f}"
        done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Preparing Data for wav2gloss"

    python3 local/data_prep.py --source ${WAV2GLOSS} --langs ${langs} --tasks ${tasks}
    for f in data/w2g_*; do
        utils/fix_data_dir.sh --utt_extra_files prev_text ${f}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
