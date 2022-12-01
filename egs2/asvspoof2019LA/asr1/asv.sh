#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# Yongyi Zang, November 2022, Modified from https://github.com/espnet/espnet/compare/master...2022fall_new_task_tutorial.

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
inference_nj=4      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
# TODO: What data_opts should be used?
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark.
fs=8k                # Sampling rate.

# asvspoof model related
asvspoof_tag=    # Suffix to the result dir for asvspoof model training.
asvspoof_config= # Config for asvspoof model training.
asvspoof_args=   # Arguments for asvspoof model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in asvspoof config.
feats_normalize=global_mvn # Normalizaton layer type.

# asvspoof related
inference_config= # Config for asvspoof model inference
inference_model=valid.acc.best.pth
inference_tag=    # Suffix to the inference dir for asvspoof model inference


# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.


help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>
Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference  # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").
    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").
    # Feature extraction related
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    # ASVSpoof model related
    --asvspoof_tag        # Suffix to the result dir for asvspoofization model training (default="${asvspoof_tag}").
    --asvspoof_config     # Config for asvspoofization model training (default="${asvspoof_config}").
    --asvspoof_args       # Arguments for asvspoofization model training, e.g., "--max_epoch 10" (default="${asvspoof_args}").
                      # Note that it will overwrite args in asvspoof config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    # ASVSpoof related
    --inference_config # Config for asvspoof model inference
    --inference_model  # asvspoofization model path for inference (default="${inference_model}").
    --inference_tag    # Suffix to the inference dir for asvspoof model inference
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set               # Name of training set (required).
    --valid_set               # Name of development set (required).
    --test_sets               # Names of evaluation sets (required).
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# # Check required arguments
# [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
# [ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
# [ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}

# Set tag for naming of model directory
if [ -z "${asvspoof_tag}" ]; then
    if [ -n "${asvspoof_config}" ]; then
        asvspoof_tag="$(basename "${asvspoof_config}" .yaml)"
    else
        asvspoof_tag="train"
    fi
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
fi

# The directory used for collect-stats mode
asvspoof_stats_dir="${expdir}/asvspoof_stats_${fs}"
# The directory used for training commands
asvspoof_exp="${expdir}/asvspoof_${asvspoof_tag}"

# ========================== Main stages start from here. ==========================
if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation."
        local/data.sh
    fi
else
    log "Skip the data preparation stages"
fi
# ========================== Data preparation is done here. ==========================
