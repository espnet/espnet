#!/bin/bash
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
SECONDS=0

# general configuration
stage=1
stop_stage=100

use_wordlm=false    # false means to train/use a character LM
lm_vocabsize=100    # effective only for word LMs
lmtag=              # tag for managing LMs
lm_resume=          # specify a snapshot file to resume LM training

# data
datadir=./downloads
an4_root=${datadir}/an4
sph2pipe=


. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

train_set="train_nodev"
train_dev="train_dev"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    if [ ! -e downloads/ ]; then
        tar -xvf downloads.tar.gz
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    mkdir -p data/{train,test}

    if [ ! -f ${an4_root}/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python local/data_prep.py ${an4_root} "${sph2pipe}"

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
    done

    # make a dev set
    utils/subset_data_dir.sh --first data/train 1 data/${train_dev}
    n=$(($(wc -l < data/train/text) - 1))
    utils/subset_data_dir.sh --last data/train ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"

