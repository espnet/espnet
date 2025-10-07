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

# General configuration
stage=1
stop_stage=100
nj=16

# For local data usage:
manifest_dir=./manifest
# For data sharing within the cluster:
# manifest_dir=/work/nvme/bbjs/shared/data_registry/manifest/LibriSpeech

# Data preparation related
train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

log "$0 $*"

# Parse options
. utils/parse_options.sh

# Set up environment variables
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

# Stage 1: Basic data preparation
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Running basic data preparation"
    log "Calling local/data.sh"

    # Run the data preparation script
    ./local/data.sh \
        --stage 1 \
        --stop_stage 3 \
        --train_set "${train_set}" \
        --train_dev "${valid_set}"

    log "Stage 1: Basic data preparation completed"
fi

# Stage 2: Prepare audio metadata and dataset JSON files
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Preparing audio metadata and dataset JSON files"
    mkdir -p ${manifest_dir}

    all_sets="${valid_set} ${test_sets} ${train_set}"
    for dataset in ${all_sets}; do
        log "Processing dataset: ${dataset}"
        mkdir -p ${manifest_dir}/${dataset}/audio1
        mkdir -p ${manifest_dir}/${dataset}/text1
        mkdir -p ${manifest_dir}/${dataset}/speaker

        python3 ../../../espnet2/speechlm/bin/prepare_audio_lhotse.py \
            --wav_scp data/${dataset}/wav.scp \
            --output_dir ${manifest_dir}/${dataset}/audio1 \
            --num_jobs ${nj}

        cp data/${dataset}/text ${manifest_dir}/${dataset}/text1/text
        cp data/${dataset}/utt2spk ${manifest_dir}/${dataset}/speaker/speaker

        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets audio1,${manifest_dir}/${dataset}/audio1,lhotse_audio \
                       text1,${manifest_dir}/${dataset}/text1/text,text \
                       speaker,${manifest_dir}/${dataset}/speaker/speaker,text \
            --output_json ${manifest_dir}/${dataset}/dataset.json
    done

    log "Stage 2 completed. Manifests in ${manifest_dir}"
fi
