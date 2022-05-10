#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${HUB4_SPANISH}
if [ -z "${HUB4_SPANISH}" ]; then
    log "Fill the value of 'HUB4_SPANISH' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
test_set="test"

audio_data=${HUB4_SPANISH}/LDC98S74
transcript_data=${HUB4_SPANISH}/LDC98T29
eval_data=${HUB4_SPANISH}/LDC2001S91
dev_list=dev.list

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
    log "stage1: Prepare eval data with ${HUB4_SPANISH}"

    # Eval dataset preparation
    # prepare_data.sh does not really care about the order or number of the
    # corpus directories
    local/prepare_data.sh \
      ${eval_data}/HUB4_1997NE/doc/h4ne97sp.sgm \
      ${eval_data}/HUB4_1997NE/h4ne_sp/h4ne97sp.sph data/${test_set}
    local/prepare_test_text.pl \
      "<unk>" data/${test_set}/text > data/${test_set}/text.clean
    mv data/${test_set}/text data/${test_set}/text.old
    mv data/${test_set}/text.clean data/${test_set}/text
    utils/fix_data_dir.sh data/${test_set}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Prepare train data with ${HUB4_SPANISH}"
    ## Training dataset preparation
    local/prepare_data.sh ${audio_data} ${transcript_data} data/${train_set}
    local/prepare_training_text.pl \
      "<unk>" data/${train_set}/text > data/${train_set}/text.clean
    mv data/${train_set}/text data/${train_set}/text.old
    mv data/${train_set}/text.clean data/${train_set}/text
    utils/fix_data_dir.sh data/${train_set}

    # For generating the dev set. Use provided utterance list otherwise
    # num_dev=$(wc -l < data/eval/segments)
    # ./utils/subset_data_dir.sh data/${train_set} ${num_dev} data/${train_dev}

    ./utils/subset_data_dir.sh --utt-list ${dev_list} data/${train_set} data/${train_dev}

    mv data/${train_set} data/${train_set}.tmp
    mkdir -p data/${train_set}
    awk '{print $1}' data/${train_dev}/segments | grep -vf - data/${train_set}.tmp/segments > data/${train_set}/uttlist.list
    ./utils/subset_data_dir.sh --utt-list data/${train_set}/uttlist.list data/${train_set}.tmp data/${train_set}
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
