#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100
fs=None
g2p=None

log "$0 $*"

. utils/parse_options.sh || exit 1;

echo "stage: $stage"

mkdir -p data

train_set="tr_no_dev"
train_dev="dev"
test_set="eval"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Set up ACE-phoneme"
    # The ACE-phoneme should be downloaded from https://github.com/timedomain-tech/ACE_phonemes.git
    # Create a soft link under mixed/svs1: `ln -s /path/to/ACE_phonemes ./ACE_phonemes`
    # |_ svs1
    #   |_ ACE_phonemes
    #   |_ svs.sh
    #   |_ ...
    # 
    #  NOTICE[IMPORTANT]: You need to add __init__.py to every directory including ACE_phonemes and its sub-directories.
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Combine Data"
    log "[IMPORTANT] assume merging with dumpped files"
    # for x in ${train_dev} ${test_set} ${train_set}; do
    #     echo "process for subset: ${x}"
    #     opts="data/${x}"
    #     for dir in $*; do
    #         if [ -d ${dir} ]; then
    #             org_workspace=$(realpath ${dir}/../../..)
    #             dataset=$(basename ${org_workspace})
    #             utils/copy_data_dir.sh ${dir}/${x} data/raw_data/"${dataset}_${x}"
    #             python local/convert_r2a_path.py ${org_workspace}/svs1 data/raw_data/"${dataset}_${x}"/wav.scp \
    #                 data/raw_data/"${dataset}_${x}"/wav.scp.tmp
    #             python local/convert_r2a_path.py ${org_workspace}/svs1 data/raw_data/"${dataset}_${x}"/score.scp \
    #                 data/raw_data/"${dataset}_${x}"/score.scp.tmp
    #             sort -o data/raw_data/"${dataset}_${x}"/wav.scp.tmp data/raw_data/"${dataset}_${x}"/wav.scp.tmp
    #             sort -o data/raw_data/"${dataset}_${x}"/score.scp.tmp data/raw_data/"${dataset}_${x}"/score.scp.tmp
    #             mv data/raw_data/"${dataset}_${x}"/wav.scp.tmp data/raw_data/"${dataset}_${x}"/wav.scp
    #             mv data/raw_data/"${dataset}_${x}"/score.scp.tmp data/raw_data/"${dataset}_${x}"/score.scp
    #             opts+=" data/raw_data/${dataset}_${x}"
    #         fi
    #     done
    #     utils/combine_data.sh --extra-files "score.scp label" ${opts}
    # done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Unifiy Phoneme-List with ACE-phoneme"
    mkdir -p score_dump
    for x in ${train_dev} ${test_set} ${train_set}; do
        echo "process for subset: ${x}"
        src_data="data/${x}"
        python local/process_phoneme.py --scp ${src_data}/score.scp
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
