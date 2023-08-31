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
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_stages=          # Spicify the stage to be skipped
skip_data_prep=false  # Skip data preparation stages.
skip_train=false      # Skip training stages.
skip_eval=false       # Skip decoding and evaluation stages.
eval_valid_set=false  # Run decoding for the validation set
ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
gpu_inference=false   # Whether to perform gpu decoding.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.
python=python3        # Specify python to execute espnet commands.
fold_length=120000     # fold_length for speech data during enhancement training

# Data preparation related
local_data_opts= # The options given to local/data.sh

# Feature extraction related
feats_type=raw_copy   # Feature type (raw, raw_copy, fbank_pitch, or extracted).
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

help_message=$(cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    stage=1               # Processes starts from the specified stage.
    stop_stage=10000      # Processes is stopped at the specified stage.
    skip_stages=          # Spicify the stage to be skipped
    skip_data_prep=false  # Skip data preparation stages.
    skip_train=false      # Skip training stages.
    skip_eval=false       # Skip decoding and evaluation stages.
    eval_valid_set=false  # Run decoding for the validation set
    ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
    num_nodes=1           # The number of nodes.
    nj=32                 # The number of parallel jobs.
    gpu_inference=false   # Whether to perform gpu decoding.
    dumpdir=dump          # Directory to dump features.
    expdir=exp            # Directory to save experiments.
    python=python3        # Specify python to execute espnet commands.
    fold_length=80000     # fold_length for speech data during enhancement training

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

EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0  ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
        exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw  ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy  ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
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
spk_stats_dir="${expdir}/spk_stats_${fs}"
if [ -z "${spk_exp}"  ]; then
    spk_exp="${expdir}/spk_${spk_tag}"
fi

# Determine which stages to skip
if "${skip_data_prep}"; then
    skip_stages+="1 2"
fi

skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# TODO (Jee-weon): add speed perturb
#if [ -n "${speed_perturb_factors}"  ]; then
#    spk_stats_dir="${spk_stats_dir}_sp"
#    spk_exp="${spk_exp}_sp"
#fi


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
    log "Stage 1 FIN."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${valid_set} ${test_sets}"
    fi

    if [ "${feats_type}" = raw ]; then
        if [ "${skip_train}" = false ]; then
            utils/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"

            # copy extra files that are not covered by copy_data_dir.sh
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${train_set}/wav.scp" "${data_feats}/${train_set}"

            echo "${feats_type}" > "${data_feats}/${train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${train_set}/audio_format"
            fi
        fi

        # Calculate EER for valid/test since speaker verification is an open set problem
        # Train can be either multi-column data or not, but valid/test always require multi-column trial
        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"

            # copy extra files that are not covered by copy_data_dir.sh
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            cp data/${dset}/trial_label "${data_feats}/${dset}"

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                --out_filename trial.scp \
                "data/${dset}/trial.scp" "${data_feats}/${dset}"
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                --out_filename trial2.scp \
                "data/${dset}/trial2.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"

        done
    elif [ "${feats_type}" = raw_copy ]; then
        if [ "${skip_train}" = false ]; then
            utils/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            echo "${feats_type}" > "${data_feats}/${train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${train_set}/audio_format"
            fi
        fi

        # Calculate EER for valid/test since speaker verification is an open set problem
        # Train can be either multi-column data or not, but valid/test always require multi-column trial
        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"
            cp data/${dset}/trial_label "${data_feats}/${dset}"
            cp data/${dset}/trial.scp "${data_feats}/${dset}"
            cp data/${dset}/trial2.scp "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"

        done
    else
        log "${feats_type} is not supported yet."
        exit 1
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Collect stats"
    _spk_train_dir="${data_feats}/${train_set}"
    _spk_valid_dir="${data_feats}/${valid_set}"

    if [ -n "${spk_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${spk_config} "
    fi

    if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    else
        # sound supports "wav", "flac", etc.
        _type=sound
    fi

    # 1. Split key file
    _logdir="${spk_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    _nj=$(min "${nj}" "$(<${_spk_train_dir}/wav.scp wc -l)" "$(<${_spk_valid_dir}/trial.scp wc -l)")

    key_file="${_spk_train_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_spk_valid_dir}/trial.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${spk_stats_dir}/run.sh'. You can resume the process from stage 3 using this script"
    mkdir -p "${spk_stats_dir}"; echo "${run_args} -- stage3 \"\$@\"; exit \$?" > "${spk_stats_dir}/run.sh"; chmod +x "${spk_stats_dir}/run.sh"

    # 3. Submit jobs
    log "Speaker collect-stats started... log: '${_logdir}/stats.*.log'"

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.spk_train \
            --collect_stats true \
            --use_preprocessor false \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,${_type} \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial.scp,speech,${_type} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --spk2utt ${_spk_train_dir}/spk2utt \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${spk_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1;  }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --skip_sum_stats --output_dir "${spk_stats_dir}"

    cp ${spk_stats_dir}/valid/speech_shape ${spk_stats_dir}/valid/speech_shape2
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Train."

    _spk_train_dir="${data_feats}/${train_set}"
    _spk_valid_dir="${data_feats}/${valid_set}"
    _opts=
    if [ -n "${spk_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${spk_config} "
    fi

    log "Spk training started... log: '${spk_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${spk_exp})"
    else
        jobname="${spk_exp}/train.log"
    fi

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${spk_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${spk_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_train \
            --use_preprocessor true \
            --resume true \
            --output_dir ${spk_exp} \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_spk_train_dir}/utt2spk,spk_labels,text \
            --train_shape_file ${spk_stats_dir}/train/speech_shape \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial.scp,speech,sound \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial2.scp,speech2,sound \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial_label,spk_labels,text \
            --spk2utt ${_spk_train_dir}/spk2utt \
            --fold_length ${fold_length} \
            --valid_shape_file ${spk_stats_dir}/valid/speech_shape \
            --output_dir "${spk_exp}" \
            ${_opts} ${spk_args}
fi
