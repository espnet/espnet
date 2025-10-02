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

# Data preparation related
train_set="train"
valid_set="dev"
test_sets="test"

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
        --data_dir "data"

    log "Stage 1: Basic data preparation completed"
fi

# Stage 2: Prepare audio metadata and dataset JSON files
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Preparing audio metadata and dataset JSON files"
    mkdir -p manifest

    all_sets="${valid_set} ${test_sets} ${train_set}"
    for dataset in ${all_sets}; do
        log "Processing dataset: ${dataset}"
        mkdir -p manifest/${dataset}/audio1 manifest/${dataset}/text1

        python3 ../../../espnet2/speechlm/bin/prepare_audio_lhotse.py \
            --wav_scp data/${dataset}/wav.scp \
            --segments data/${dataset}/segments \
            --output_dir manifest/${dataset}/audio1 \
            --num_jobs 32

        cp data/${dataset}/text manifest/${dataset}/text1/text

        python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets audio1,manifest/${dataset}/audio1/cuts.jsonl.gz,lhotse_audio \
                       text1,manifest/${dataset}/text1/text,text \
            --output_json manifest/${dataset}/dataset.json
    done

    log "Stage 2 completed. Manifests in ./manifest/"
fi