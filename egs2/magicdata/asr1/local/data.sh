#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

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

# TODO: auto download and:
#   1. mkdir wav
#   2. put train, test, dev subset under wav/
#   3. copy train.scp, test,scp, dev.scp into wav/train/wav.scp, wav/test/wav.scp, wav/dev/wav.scp
if [ -z "${MAGICDATA}" ]; then
  log "Error: \$MAGICDATA is not set in db.sh."
  log "You may request the MAGICDATA dataset from http://www.openslr.org/68/"
  exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"

    local/prepare_data.sh ${MAGICDATA}/wav/train data/local/train data/train || exit 1;
    local/prepare_data.sh ${MAGICDATA}/wav/test data/local/test data/test || exit 1;
    local/prepare_data.sh ${MAGICDATA}/wav/dev data/local/dev data/dev || exit 1;

    # Normalize text to capital letters
    for x in train dev test; do
        mv data/${x}/text data/${x}/text.org
        paste <(cut -f 1 data/${x}/text.org) <(cut -f 2 data/${x}/text.org | tr '[:lower:]' '[:upper:]') \
            > data/${x}/text
        rm data/${x}/text.org
    done

    echo "Exclude train utterances with English tokens. "
    echo "Set \$train_set to train in run.sh if you don't want this" 
    mkdir -p data/train_noeng 
    cp data/train/{wav.scp,spk2utt,utt2spk} data/train_noeng
    awk 'eng=0;{for(i=2;i<=NF;i++)if($i ~ /^.*[A-Z]+.*$/)eng=1}{if(eng==0)print $0}' data/train/text > data/train_noeng/text
    utils/fix_data_dir.sh data/train_noeng/
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
