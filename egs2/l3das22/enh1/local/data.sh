#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


help_message=$(cat << EOF
Usage: $0 
  optional argument:
    None
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


stage=1
stop_stage=2

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${L3DAS22}" ] ; then
    log "
    Please fill the value of 'L3DAS22' in db.sh
    The 'L3DAS22' (https://www.kaggle.com/l3dasteam/l3das22) directory 
    should at least contain the task1 dataset:
        L3DAS22
        ├── L3DAS22_Task1_dev
        ├── L3DAS22_Task1_test
        ├── L3DAS22_Task1_train100
        ├── L3DAS22_Task1_train360_1
        └── L3DAS22_Task1_train360_2
    "
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Data preparation stage 1: Multi-Channel data preparation"
    # The following datasets will be created:
    # L3DAS22_Task1_dev, L3DAS22_Task1_test, L3DAS22_Task1_train100, L3DAS22_Task1_train360
    local/l3das22_multi_channel.sh ${L3DAS22} || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Data preparation stage 2: Datasets preparation"
    # The following directories will be created:
    # train_multich, dev_multich, test_multich
    local/l3das22_data_prep.sh  ${L3DAS22} || exit 1;
fi

