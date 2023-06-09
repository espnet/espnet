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
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_stages=          # Spicify the stage to be skipped
skip_data_prep=false  # Skip data preparation stages.
skip_train=false      # Skip training stages.
skip_eval=false       # Skip decoding and evaluation stages.
skip_upload=true      # Skip packing and uploading to zenodo
skip_upload_hf=true   # Skip uploading to hugging face stages.
eval_valid_set=false  # Run decoding for the validation set
n_gpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
inference_nj=32       # The number of parallel jobs in decoding.
gpu_inference=false   # Whether to perform gpu decoding.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.

run_args=$(scripts/utils/print_args.sh $0 "$@")

local_data_opts=
n_train_frame=
n_eval_frame=

. utils/parse_options.sh

if [ $# -ne 0  ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
        exit 2
fi

. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
    log "Stage 1 FIN."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 2: Train."
    ${python} -m espnet2.bin.launch \
        --cmd ${cuda_cmd} --name ${jobname} \
        --log ${sv_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${sv_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_train \
            --use_preprocessor true \
            --resume true \
            --output_dir ${sv_exp} \
            ${_opts} ${sv_args}
fi
