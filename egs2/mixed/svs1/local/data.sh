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
combine_path=None

log "$0 $*"

. utils/parse_options.sh || exit 1;

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
    IFS='$' read -r -a datasets_path <<< "$combine_path"
    if [ -e "data/raw_data" ]; then
        echo "delete data/raw_data"
        rm -r "data/raw_data"
    fi
    for x in ${train_dev} ${test_set} ${train_set}; do
        echo "process for subset: ${x}"
        opts="data/${x}"
        for dir in "${datasets_path[@]}"; do
            if [ -d ${dir} ]; then
                org_workspace=$(realpath ${dir}/../../..)
                dataset=$(basename ${org_workspace})
                echo $dir
                utils/copy_data_dir.sh ${dir}/${x} data/raw_data/"${dataset}_${x}"
                python local/convert_r2a_path.py ${org_workspace}/svs1 data/raw_data/"${dataset}_${x}"/wav.scp \
                    data/raw_data/"${dataset}_${x}"/wav.scp.tmp
                python local/convert_r2a_path.py ${org_workspace}/svs1 data/raw_data/"${dataset}_${x}"/score.scp \
                    data/raw_data/"${dataset}_${x}"/score.scp.tmp
                sort -o data/raw_data/"${dataset}_${x}"/wav.scp.tmp data/raw_data/"${dataset}_${x}"/wav.scp.tmp
                sort -o data/raw_data/"${dataset}_${x}"/score.scp.tmp data/raw_data/"${dataset}_${x}"/score.scp.tmp
                mv data/raw_data/"${dataset}_${x}"/wav.scp.tmp data/raw_data/"${dataset}_${x}"/wav.scp
                mv data/raw_data/"${dataset}_${x}"/score.scp.tmp data/raw_data/"${dataset}_${x}"/score.scp
                opts+=" data/raw_data/${dataset}_${x}"
            else
                echo "Dataset dicretory ${dir} does not exist."
                exit 1
            fi
        done
        utils/combine_data.sh --extra-files "score.scp label utt2spk" ${opts}
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Resample to ${fs}Hz if needed"
    for x in ${train_dev} ${test_set} ${train_set}; do
        echo "Process for subset: ${x}"
        src_data="data/${x}"
        dst_wav_dump_dir="wav_dump_resampled/${fs}Hz"
        mv ${src_data}/wav.scp ${src_data}/wav.scp.tmp
        ./local/resample_wav_scp.sh ${fs} ${src_data}/wav.scp.tmp ${dst_wav_dump_dir} ${src_data}/wav.scp
        rm ${src_data}/wav.scp.tmp
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 3: Unifiy Phoneme-List with ACE-phoneme"
    mkdir -p score_dump
    for x in ${train_dev} ${test_set} ${train_set}; do
        echo "process for subset: ${x}"
        src_data="data/${x}"
        python local/process.py --scp ${src_data}
        mv ${src_data}/score.scp.tmp ${src_data}/score.scp
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/text.tmp ${src_data}/text
        mv ${src_data}/utt2lang.tmp ${src_data}/utt2lang
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
