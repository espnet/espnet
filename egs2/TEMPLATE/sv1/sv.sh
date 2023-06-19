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

SECONDS=0

# General configuration
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_stages=          # Spicify the stage to be skipped
skip_data_prep=false  # Skip data preparation stages.
skip_train=false      # Skip training stages.
skip_eval=false       # Skip decoding and evaluation stages.
eval_valid_set=false  # Run decoding for the validation set
n_gpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
gpu_inference=false   # Whether to perform gpu decoding.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.
python=python3        # Specify python to execute espnet commands.

run_args=$(scripts/utils/print_args.sh $0 "$@")

# Feature extraction related
feats_type=raw       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.
min_wav_duration=1.0  # Minimum duration in second.
max_wav_duration=60.  # Maximum duration in second.

# Speaker model related
spk_exp=              # Specify the directory path for spk experiment.
spk_tag=              # Suffix to the result dir for spk model training.
spk_config=           # Config for the spk model training.
spk_args=             # Arguments for spk model training.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

. utils/parse_options.sh

if [ $# -ne 0  ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
        exit 2
fi

. ./path.sh
. ./cmd.sh
echo "${stage} ${stop_stage}"

# Check feature type
if [ "${feats_type}" = raw  ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy  ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
elif [ "${feats_type}" = fbank_pitch  ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank  ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted  ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${spk_tag}" ]; then
    if [ -n "${spk_config}" ]; then
        spk_tag="$(basename "${spk_config}" .yaml)_${feats_type}"
    else
        spk_tag="train_${feats_type}"
    fi
fi

# Set directory used for training commands
if [ -z "${spk_exp}"  ]; then
    spk_exp="${expdir}/spk_${spk_tag}"
fi


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
    log "Stage 1 FIN."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi

    if [ "${feats_type}" = raw ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}"
        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}"]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done
    else
        log "${feats_type} is not supported yet."
        exit 1
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Train."
    ${python} -m espnet2.bin.launch \
        --cmd ${cuda_cmd} --name ${jobname} \
        --log ${spk_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${sv_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_train \
            --use_preprocessor true \
            --resume true \
            --output_dir ${spk_exp} \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_spk_train_dir}/utt2spk,spk_labels,text \
            ${_opts} ${spk_args}
fi

