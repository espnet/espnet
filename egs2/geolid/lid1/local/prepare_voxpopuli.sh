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


stage=1
stop_stage=100
dataset_name="voxpopuli"
train_set="train_${dataset_name}_lang"
valid_set="dev_${dataset_name}_lang"
test_set="test_${dataset_name}_lang"
dataset_path="downloads/voxpopuli"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

partitions="${train_set} ${valid_set} ${test_set}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: download voxpopuli"
    log "skipping for now"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: prepare voxpopuli"
    python local/prepare_voxpopuli.py --dataset_path ${dataset_path}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: make utt2lang to lang2utt"
    for x in ${partitions}; do
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2lang > data/${x}/lang2utt
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: validate data"
    for x in ${partitions}; do
        mv data/${x}/utt2lang data/${x}/utt2spk
        mv data/${x}/lang2utt data/${x}/spk2utt
        utils/fix_data_dir.sh "data/${x}" || exit 1;
        utils/validate_data_dir.sh --no-feats "data/${x}" --no-text
        mv data/${x}/utt2spk data/${x}/utt2lang
        mv data/${x}/spk2utt data/${x}/lang2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
