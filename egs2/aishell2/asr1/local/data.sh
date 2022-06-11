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

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh
. ./db.sh


if [ -z "${AISHELL2}" ]; then
  log "Error: \$AISHELL2 is not set in db.sh."
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    # For training set
    local/prepare_data.sh ${AISHELL2}/iOS/data data/local/train data/train || exit 1;

    # # For dev and test set
    for x in Android iOS Mic; do
        local/prepare_data.sh ${AISHELL2}/AISHELL-DEV-TEST-SET/${x}/dev data/local/dev_${x,,} data/dev_${x,,} || exit 1;
        local/prepare_data.sh ${AISHELL2}/AISHELL-DEV-TEST-SET/${x}/test data/local/test_${x,,} data/test_${x,,} || exit 1;
    done

    # Normalize text to capital letters
    for x in train dev_android dev_ios dev_mic test_android test_ios test_mic; do
        mv data/${x}/text data/${x}/text.org
        paste <(cut -f 1 data/${x}/text.org) <(cut -f 2 data/${x}/text.org | tr '[:lower:]' '[:upper:]') \
            > data/${x}/text
        rm data/${x}/text.org
    done

    echo "Exclude train utterances with English tokens. "
    echo "Set train_set to train if you don't want this" 
    mkdir -p data/train_noeng 
    cp data/train/{wav.scp,spk2utt,utt2spk} data/train_noeng
    cat data/train/text | awk 'eng=0;{for(i=2;i<=NF;i++)if($i ~ /^.*[A-Z]+.*$/)eng=1}{if(eng==0)print $0}' > data/train_noeng/text
    utils/fix_data_dir.sh data/train_noeng/

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
