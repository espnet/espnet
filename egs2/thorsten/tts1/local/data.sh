#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=0

valid_size=100
test_size=100
seed=100

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${THORSTEN}" ]; then
    log "Fill the value of 'THORSTEN' of db.sh"
    exit 1
fi
db_root=${THORSTEN}
corpus_root="${db_root}/ThorstenVoice-Dataset_2022.10"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Preparation"

    rm -rf data/train data/valid data/test

    # The released metadata is split into train/dev/test, but the dev set holds
    # only 4 utterances. local/prepare_data.py therefore merges the three files
    # and re-splits them deterministically so that the split is reproducible.
    python3 local/prepare_data.py \
        --db-root "${corpus_root}" \
        --output-root data \
        --valid-size "${valid_size}" \
        --test-size "${test_size}" \
        --seed "${seed}"

    for split in train valid test; do
        utils/utt2spk_to_spk2utt.pl \
            < "data/${split}/utt2spk" \
            > "data/${split}/spk2utt"

        utils/validate_data_dir.sh --no-feats "data/${split}"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
