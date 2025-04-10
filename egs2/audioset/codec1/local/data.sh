#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--use_unbal <true/false>] [--dev_ratio <0-1>] [--seed <int>]
Options:
    --use_unbal    # Whether to use unbalanced data (default: false)
    --dev_ratio    # Ratio of development set (default: 0.02)
    --seed         # Random seed for splitting (default: 777)
EOF
)

. ./db.sh
. ./path.sh

use_unbal=false
dev_ratio=0.02
seed=777
stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "${AUDIOSET}" ]; then
    log "Fill the value of 'AUDIOSET' of db.sh"
    exit 1
fi

download_dir=${AUDIOSET}
data_dir=data
base_url=https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data

audioset_complete_file="${download_dir}/.audioset_complete"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -e "${audioset_complete_file}" ]; then
        log "AudioSet data download and extraction already done. Skipped."
    else
        log "stage 0: Downloading and extracting AudioSet data"
        mkdir -p ${download_dir}

        files=()
        for i in {00..09}; do files+=("bal_train${i}.tar"); done
        for i in {00..08}; do files+=("eval${i}.tar"); done
        if ${use_unbal}; then
            for i in {000..869}; do files+=("unbal_train${i}.tar"); done
        fi

        for f in "${files[@]}"; do
            url="${base_url}/${f}"
            tar_path="${download_dir}/${f}"
            if [ ! -f ${tar_path} ]; then
                log "Downloading ${url}"
                wget -q --show-progress -O ${tar_path} ${url}
            fi

            log "Extracting ${tar_path}"
            tar -xf ${tar_path} -C ${download_dir}
        done

        mkdir -p ${data_dir}/audioset/test/raw ${data_dir}/audioset/train/raw
        find -L ${download_dir}/audio/eval -name "*.flac" -exec mv {} ${data_dir}/audioset/test/raw \;
        find -L ${download_dir}/audio/bal_train -name "*.flac" -exec mv {} ${data_dir}/audioset/train/raw \;
        if ${use_unbal}; then
            find -L ${download_dir}/audio/unbal_train -name "*.flac" -exec mv {} ${data_dir}/audioset/train/raw \;
        fi

        touch "${audioset_complete_file}"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Converting flac to raw wav"

    for dset in test train; do
        raw_dir=${data_dir}/audioset/${dset}/raw
        wav_dir=${data_dir}/audioset/${dset}/wav

        mkdir -p ${wav_dir}
        find -L ${raw_dir} -name "*.flac" | while read -r flac_path; do
            wav_path=${wav_dir}/$(basename ${flac_path%.*}).wav
            if [ ! -f ${wav_path} ]; then
                sox ${flac_path} ${wav_path}
            fi
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Standardized to 16khz mono"

    for dset in test train; do
        wav_dir=${data_dir}/audioset/${dset}/wav
        norm_dir=${data_dir}/audioset/${dset}/normed_wav

        mkdir -p ${norm_dir}
        find -L ${wav_dir} -name "*.wav" | while read -r raw_wav; do
            norm_wav=${norm_dir}/$(basename ${raw_wav})
            if [ ! -f ${norm_wav} ]; then
                sox ${raw_wav} -r 16000 -c 1 -b 16 ${norm_wav} remix 1 rate -v dither -s
            fi
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data preparation"

    for dset in test train; do

        norm_dir=$(realpath "${data_dir}/audioset/${dset}/normed_wav")
        if [ "$dset" = "train" ]; then
            dest_dir="${data_dir}/${dset}_all"
        else
            dest_dir="${data_dir}/${dset}"
        fi
        mkdir -p "${dest_dir}"

        find -L "${norm_dir}" -name "*.wav" | \
        awk -F'/' -v OFS=' ' '{
            filename = $NF;
            id = substr(filename, 1, length(filename)-4);
            print id, $0
        }' > "${dest_dir}/wav.scp"

        awk '{print $1, $1}' "${dest_dir}/wav.scp" > "${dest_dir}/utt2spk"
        utils/utt2spk_to_spk2utt.pl "${dest_dir}/utt2spk" > "${dest_dir}/spk2utt"
    done

    utils/shuffle_list.pl --srand ${seed} ${data_dir}/train_all/wav.scp > ${data_dir}/train_all/shuffled.scp
    num_total=$(wc -l < ${data_dir}/train_all/shuffled.scp)
    num_dev=$(printf "%.0f" "$(echo "${num_total} * ${dev_ratio}" | bc)")

    mkdir -p ${data_dir}/dev ${data_dir}/train
    head -n ${num_dev} ${data_dir}/train_all/shuffled.scp > ${data_dir}/dev/wav.scp
    tail -n +$((num_dev+1)) ${data_dir}/train_all/shuffled.scp > ${data_dir}/train/wav.scp

    for dset in train dev; do
        awk '{print $1, $1}' ${data_dir}/${dset}/wav.scp > ${data_dir}/${dset}/utt2spk
        utils/utt2spk_to_spk2utt.pl ${data_dir}/${dset}/utt2spk > ${data_dir}/${dset}/spk2utt
    done
    test_wav_scp="${data_dir}/test/wav.scp"
    test_shuffled_scp="${data_dir}/test/shuffled_wav.scp"
    utils/shuffle_list.pl --srand ${seed} "$test_wav_scp" > "$test_shuffled_scp"

    test_subset_wav_scp="${data_dir}/test_subset/wav.scp"
    mkdir -p "${data_dir}/test_subset"
    head -n 1000 "$test_shuffled_scp" > "$test_subset_wav_scp"

    awk '{print $1, $1}' "$test_subset_wav_scp" > "${data_dir}/test_subset/utt2spk"
    utils/utt2spk_to_spk2utt.pl "${data_dir}/test_subset/utt2spk" > "${data_dir}/test_subset/spk2utt"

    utils/validate_data_dir.sh --no-text --no-feats "${data_dir}/test_subset"

    rm "$test_shuffled_scp"
    for dset in dev train test test_subset; do
        utils/validate_data_dir.sh --no-text --no-feats ${data_dir}/${dset}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
