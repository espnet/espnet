#!/usr/bin/env bash

# set -e
# set -u
# set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=2


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# the downloaded data is already processed in kaldi format, named as data/ after extracting
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -f "iban.tar.gz" ]; then
        wget https://www.openslr.org/resources/24/iban.tar.gz
        log "data stage 1: finish downloading iban.tar.gz"
    else
        log "data stage 1: iban.tar.gz already exists. Skip data downloading."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ! -d "data" ]; then
        # extract and format
        tar -xvf iban.tar.gz; rm iban.tar.gz
        mkdir data/docs
        mv IS2015_samsonjuan.pdf data/docs/
        mv LICENSE.html data/docs/
        mv README data/docs/
        sed -i 's|asr_iban/||g' data/train/wav.scp
        sed -i 's|asr_iban/||g' data/dev/wav.scp
        # make test set: move ib{m,f}_010 from train to test
        mkdir -p data/test
        for file in wav.scp text utt2spk spk2utt; do
            grep "_010_" data/train/${file} > data/test/${file}
            sed -i '/_010_/d' data/train/${file}
        done
        log "data stage 2: Data preparation completed."
    else
        log "data stage 2: data/ already exists. Skip data preparation."
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
