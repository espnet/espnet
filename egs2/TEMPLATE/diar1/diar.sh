#!/bin/bash

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
stage=1          # Processes starts from the specified stage.
stop_stage=10000 # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
inference_nj=32     # The number of parallel jobs in decoding.
gpu_inference=false # Whether to perform gpu decoding.
expdir=exp       # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=8k             # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second

# diar model related
diar_tag=    # Suffix to the result dir for diar model training.
diar_config= # Config for diar model training.
diar_args=   # Arguments for diar model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in diar config.
spk_num=2
total_spk_num=

# diar related
inference_args="--normalize_output_wav true"
inference_model=valid.acc.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
diar_speech_fold_length=800 # fold_length for speech data during diar training
lang=noinfo      # The language type of corpus


help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu          # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes     # The number of nodes
    --nj            # The number of parallel jobs (default="${nj}").
    --inference_nj  # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir       # Directory to dump features (default="${dumpdir}").
    --expdir        # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type   # Feature type (only support raw currently).
    --audio_format # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").


    # Diarization model related
    --diar_tag    # Suffix to the result dir for diarization model training (default="${diar_tag}").
    --diar_config # Config for diarization model training (default="${diar_config}").
    --diar_args   # Arguments for diarization model training, e.g., "--max_epoch 10" (default="${diar_args}").
                 # Note that it will overwrite args in diar config.
    --spk_num    # Number of speakers in the input audio (default="${spk_num}")
    --total_spk_num # Total number fo speakers, necessary for EEND loss (default="${total_spk_num})

    # diarization related
    --inference_args      # Arguments for diarization in the inference stage (default="${inference_args}")
    --inference_model # diarization model path for inference (default="${inference_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --diar_speech_fold_length # fold_length for speech data during diarization training  (default="${diar_speech_fold_length}").
    --lang         # The language type of corpus (default="${lang}")
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


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw

# Set tag for naming of model directory
if [ -z "${diar_tag}" ]; then
    if [ -n "${diar_config}" ]; then
        diar_tag="$(basename "${diar_config}" .yaml)_${feats_type}"
    else
        diar_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${diar_args}" ]; then
        diar_tag+="$(echo "${diar_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for collect-stats mode
diar_stats_dir="${expdir}/diar_stats_${fs}"
# The directory used for training commands
diar_exp="${expdir}/diar_${diar_tag}"

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

        log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{wav.scp,reco2file_and_channel}

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}"  \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            # specifics for diarization
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}${_suf}/${dset}"/utt2spk \
                    "${data_feats}${_suf}/${dset}"/segments \
                    "${data_feats}${_suf}/${dset}"/rttm
        done
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            # diarization typically accept long recordings, so does not has
            # max length requirements
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" \
                    '{ if ($2 > min_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"
            
            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"

            # specifics for diarization
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}/${dset}"/utt2spk \
                    "${data_feats}/${dset}"/segments \
                    "${data_feats}/${dset}"/rttm
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _diar_train_dir="${data_feats}/${train_set}"
        _diar_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: Diarization collect stats: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

        _opts=
        if [ -n "${diar_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
            _opts+="--config ${diar_config} "
        fi

        _feats_type="$(<${_diar_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        if [ -z "${total_spk_num}" ]; then
            # Training speaker numbers
            total_spk_num=$(wc -l <${_diar_train_dir}/spk2utt)
        fi
        _opts+="--total_spk_num ${total_spk_num} "

        # 1. Split the key file
        _logdir="${diar_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_diar_train_dir}/${_scp} wc -l)" "$(<${_diar_valid_dir}/${_scp} wc -l)")

        key_file="${_diar_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${diar_stats_dir}/run.sh'. You can resume the process from stage 9 using this script"
        mkdir -p "${diar_stats_dir}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${diar_stats_dir}/run.sh"; chmod +x "${diar_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Diarization collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.diar_train \
                --collect_stats true \
                --use_preprocessor true \
                --train_data_path_and_name_and_type "${_diar_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_diar_train_dir}/rttm,spk_labels,rttm" \
                --valid_data_path_and_name_and_type "${_diar_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_diar_train_dir}/rttm,spk_labels,rttm" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${diar_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${diar_stats_dir}"

    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _diar_train_dir="${data_feats}/${train_set}"
        _diar_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: Diarization Training: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

        _opts=
        if [ -n "${diar_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
            _opts+="--config ${diar_config} "
        fi

        _feats_type="$(<${_diar_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((diar_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${diar_stats_dir}/train/feats_stats.npz "
        fi

        _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/rttm,spk_labels,rttm "
        _opts+="--train_shape_file ${diar_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${diar_stats_dir}/train/rttm_shape "

        log "Generate '${diar_exp}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${diar_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${diar_exp}/run.sh"; chmod +x "${diar_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "Diarization training started... log: '${diar_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${diar_exp})"
        else
            jobname="${diar_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${diar_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${diar_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.diar_train \
                --use_preprocessor true \
                --valid_data_path_and_name_and_type "${_diar_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_diar_valid_dir}/text,text,text" \
                --valid_shape_file "${diar_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${diar_stats_dir}/valid/text_shape.${token_type}" \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${diar_text_fold_length}" \
                --output_dir "${diar_exp}" \
                ${_opts} ${diar_args}

    fi
else
    log "Skip the training stages"
fi
