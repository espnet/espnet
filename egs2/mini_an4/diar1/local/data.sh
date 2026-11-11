#!/usr/bin/env bash
# Two-speaker mixtures of mini_an4 utterances for the diarization CI test.
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
an4_root=./downloads/an4

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Untar downloads.tar.gz"
    if [ ! -e downloads/ ]; then
        tar -xf downloads.tar.gz
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Simulate two-speaker mixtures"
    if [ ! -f ${an4_root}/README ]; then
        log "Cannot find an4 root ${an4_root}"
        exit 1
    fi
    # train: every pair of speakers of an4_clstk; test (also the validation
    # set): the two speakers of an4test_clstk
    python3 local/simulate_mixtures.py "${an4_root}/wav" an4_clstk \
        data/mixtures/train data/train
    python3 local/simulate_mixtures.py --seed 1 "${an4_root}/wav" an4test_clstk \
        data/mixtures/test data/test
    # VAD (num_spk 1): the same mixtures, labelled with the union of the speech
    python3 local/simulate_mixtures.py --vad "${an4_root}/wav" an4_clstk \
        data/mixtures/train data/train_vad
    python3 local/simulate_mixtures.py --vad --seed 1 "${an4_root}/wav" an4test_clstk \
        data/mixtures/test data/test_vad
    for x in train test train_vad test_vad; do
        utils/validate_data_dir.sh --no-feats --no-text data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
