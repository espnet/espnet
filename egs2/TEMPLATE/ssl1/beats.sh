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
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
datadir=data         # Directory to save the prepared data from Stage 1.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw). 
                    # flac does not work with kaldi during audio tokenization phase
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Pretrain model related
ssl_tag=       # Suffix to the result dir for ssl model training.
beats_args=         # Arguments for ssl model training, e.g., "--max_epoch 10".
                     # Note that it will overwrite args in ssl config.
num_splits_ssl=1 # Number of splitting for corpus.

# Pretrain related
train_start_iter=0 # Pretrain starts from the specified iteration
train_stop_iter=2  # Pretrain is stopped from the specified iteration
train_config=    # Configration file of training stage
n_targets=             # Number of codebook targets
tokenizer_inference_batch_size=32 # Batch size for tokenizer inference
tokenizer_train_config= # Configration file of tokenizer training stage
tokenizer_inference_config= # Configration file of tokenizer inference stage
external_tokenizer_model= # External tokenizer model to use for tokenizer inference
external_teacher_model= # External teacher model to use for tokenizer inference

# Upload model related
inference_ssl_model=valid.loss.best.pth # SSL model path from previous iteration and uploading
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=     # Name of training set
valid_set=     # Name of valid set

speech_fold_length=160000 # fold_length for speech data during SSL training.
text_fold_length=500   # fold_length for text data during SSL training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}")
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --datadir        # Directory to save the prepared data from Stage 1 (default="${datadir}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --ssl_tag        # Suffix to the result dir for ssl model training (default="${ssl_tag}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (raw, or fbank, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").


    # BEATs model related
    --num_splits_ssl   # Number of splitting for ssl training  (default="${num_splits_ssl}").

    # BEATs related
    --train_start_iter  # Pretrain starts from the specified iteration (0 mean MFCC iteraion, default="${train_start_iter}").
    --train_stop_iter   # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion, default="${train_stop_iter}").
    --train_config    # configration file of training stage
    --n_targets       # number of codebook vectors
    --tokenizer_inference_batch_size # Batch size for tokenizer inference (default="${tokenizer_inference_batch_size}").
    --tokenizer_train_config # configration file of tokenizer training stage
    --tokenizer_inference_config # Configration file of tokenizer inference stage.
    --external_tokenizer_model  # External tokenizer model to use for tokenizer inference.
    --external_teacher_model    # External teacher model to use for tokenizer inference.
    --beats_args      # Arguments for beats model training (default="${beats_args}").
                       # e.g., --beats_args "--max_epoch 10"
                       # Note that it will overwrite args in pt config.

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training train set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).

    --speech_fold_length  # fold_length for speech data during BEATs training (default="${speech_fold_length}").
    --text_fold_length    # fold_length for text data during BEATs training (default="${text_fold_length}").
    --inference_ssl_model # SSL model path from previous iteration and uploading (default="${inference_ssl_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
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
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi
data_feats_raw=${dumpdir}/raw
ssl_stats_dir="${expdir}/beats_stats_${feats_type}"

if ! [ ${train_start_iter} -le ${train_stop_iter} ]; then
    log "Error: train_start_iter is required to be smaller or equal than train_stop_iter"
fi

token_listdir="${datadir}/token_list_${n_targets}codebook"
mkdir -p "${token_listdir}"

if [ -z "${ssl_tag}" ]; then
    if [ -n "${train_config}" ]; then
        ssl_tag="$(basename "${train_config}" .yaml)_${feats_type}"
    else
        ssl_tag="train_${feats_type}"
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for ${datadir}/${train_set}, ${datadir}/${valid_set}, etc."
        local/data.sh ${datadir} ${local_data_opts}
    fi

    if ! [[ "${feats_type}" = fbank || "${feats_type}" = raw ]]; then
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(shikhar): Add support for segment file
        log "Stage 2: Format wav.scp: ${datadir}/ -> ${data_feats_raw}/[org]"
        for dset in "${valid_set}" "${train_set}" ; do
            utils/copy_data_dir.sh --validate_opts --non-print ${datadir}/"${dset}" "${data_feats_raw}/org/${dset}"
            rm -f ${data_feats_raw}/org/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                                            --audio-format "${audio_format}" --fs "${fs}" \
                                            "${datadir}/${dset}/wav.scp" "${data_feats_raw}/org/${dset}"

            # Copy data from multiple jobs
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats_raw}/org/${dset}" "${data_feats_raw}/${dset}"
            echo "raw" > "${data_feats_raw}/${dset}/feats_type"
        done
        
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats_raw}"

        for dset in "${train_set}" "${valid_set}"; do
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats_raw}/org/${dset}/utt2num_samples" \
            awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
            '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
            >"${data_feats_raw}/${dset}/utt2num_samples"
            
            <"${data_feats_raw}/org/${dset}/wav.scp" \
            utils/filter_scp.pl "${data_feats_raw}/${dset}/utt2num_samples"  \
            >"${data_feats_raw}/${dset}/wav.scp"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats_raw}/${dset}"
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        if [ "${feats_type}" = fbank ]; then
            log "Stage 4: Feature extraction: ${data_feats}/${train_set}, ${data_feats}/${valid_set}"
            for dset in "${valid_set}" "${train_set}"; do
                log "Extracting features: ${dset}"
                utils/copy_data_dir.sh --validate_opts --non-print "${dumpdir}/raw/${dset}" "${data_feats}/org/${dset}"
                rm -f ${data_feats}/org/${dset}/{segments,reco2file_and_channel,reco2dur}
                # shellcheck disable=SC2086
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                steps/make_fbank.sh --cmd "${train_cmd}" --nj "${nj}" --fs "${_fs}" --fbank_stats_file fbank_stats \
                              --n_mels 128 --use_kaldi true "${data_feats}/org/${dset}"

                echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"
                utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
                cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
                cp "${data_feats}/org/${dset}/fbank_stats.txt" "${data_feats}/${dset}/fbank_stats.txt"
                # NOTE(shikhar): After this stage we have data in both raw and fbank format.
            done
        else
            log "Skip the stage for feature extraction"
        fi
    fi
else
    log "Skip the stages for data preparation"
fi


setup_common_vars() {
    _ssl_train_dir="${data_feats}/${train_set}"
    _ssl_valid_dir="${data_feats}/${valid_set}"
    
    # Set tokenizer tag
    _tokenizer_inference_tag="tok"
    if [ -n "${tokenizer_inference_config}" ]; then
        _tokenizer_inference_tag="$(basename "${tokenizer_inference_config}" .yaml)"
    fi
    
    # Determine file types
    _feats_type="$(<${_ssl_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        _type=sound
        _waveform_opt="--waveform_input true"
    elif [ "${_feats_type}" = fbank ]; then
        _scp=feats.scp
        _type=kaldi_ark
        _waveform_opt="--waveform_input false"
    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
}


generate_checkpoint() {
    run_dir_=$1
    output_path_=$2

    latest_checkpoint_dir_=$(find "${run_dir_}/checkpoint_"* -type d | sort -V | tail -n 1)
    if [[ -z "$latest_checkpoint_dir_" ]]; then
        log "Error: No checkpoints found in ${run_dir_}"
        return 1
    fi
    checkpoint_num_=$(basename "${latest_checkpoint_dir_}" | grep -oE '[0-9]+')
    if [[ -z "$checkpoint_num_" ]]; then
        log "Error: Failed to extract checkpoint number from ${latest_checkpoint_dir_}"
        return 1
    fi

    ${python} espnet2/beats/generate_beats_checkpoint.py \
        --espnet_model_checkpoint_path "${latest_checkpoint_dir_}/${checkpoint_num_}/mp_rank_00_model_states.pt" \
        --output_path "${output_path_}" \
        --espnet_model_config_path "${run_dir_}/config.yaml" \
        --deepspeed_checkpoint

    log "Checkpoint converted and stored at ${output_path_}"
}

train_encoder() {
    iteration=$1
    ssl_exp="${expdir}/beats_iter${iteration}_${ssl_tag}"
    
    log "Training encoder for iteration ${iteration}..."
    
    _opts=""
    [ -n "${train_config}" ] && _opts+="--config ${train_config} "

    if [ "${num_splits_ssl}" -gt 1 ]; then
        _split_dir="${ssl_stats_dir}/splits${num_splits_ssl}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
                --scps "${_ssl_train_dir}/${_scp}" \
                "${_ssl_train_dir}/target_iter${iteration}_${_tokenizer_inference_tag}" \
                "${ssl_stats_dir}/train/speech_shape" \
                "${ssl_stats_dir}/train/target_shape.word" \
                --num_splits "${num_splits_ssl}" \
                --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text "
        _opts+="--train_shape_file ${_split_dir}/speech_shape "
        _opts+="--train_shape_file ${_split_dir}/target_shape.word "
        _opts+="--multiple_iterator true "
    else
        _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text "
        _opts+="--train_shape_file ${ssl_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${ssl_stats_dir}/train/target_shape.word "
    fi

    mkdir -p "${ssl_exp}"
    echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${ssl_exp}/run.sh"
    chmod +x "${ssl_exp}/run.sh"

    jobname=$(echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl && basename ${ssl_exp} || echo "${ssl_exp}/train.log")
    
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${ssl_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${ssl_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.beats_train \
            --use_preprocessor true \
            --token_list "${token_listdir}/tokens.txt" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text" \
            --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${ssl_stats_dir}/valid/target_shape.word" \
            --resume true \
            --fold_length "${speech_fold_length}" \
            --fold_length "${text_fold_length}" \
            --output_dir "${ssl_exp}" \
            ${_opts} ${beats_args}
    
    # Generate float32 checkpoint after training completes
    checkpoint_path="${ssl_exp}/epoch_latest.pt"
    log "Generating float32 checkpoint from encoder training: ${checkpoint_path}"
    generate_checkpoint "${ssl_exp}" "${checkpoint_path}"
}

train_tokenizer() {
    iteration=$1
    ssl_tokenizer_exp="${expdir}/beats_tokenizer_iter${iteration}_${ssl_tag}"
    
    _opts=""
    [ -n "${tokenizer_train_config}" ] && _opts+="--config ${tokenizer_train_config} "
    
    # Setup teacher
    prev_iter=$((iteration - 1))
    prev_model_dir="${expdir}/beats_iter${prev_iter}_${ssl_tag}"
    teacher_ckpt_path_="${prev_model_dir}/epoch_latest.pt"

    if [ -n "${external_teacher_model}" ]; then
        teacher_ckpt_path_="${external_teacher_model}"
    else
        if [ ! -f "${teacher_ckpt_path_}" ]; then
            log "Generating teacher checkpoint from previous iteration"
            generate_checkpoint "${prev_model_dir}" "${teacher_ckpt_path_}"
        fi
    fi

    log "Training tokenizer for iteration ${iteration} using teacher: ${teacher_ckpt_path_}"

    mkdir -p "${ssl_tokenizer_exp}"
    echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${ssl_tokenizer_exp}/run.sh"
    chmod +x "${ssl_tokenizer_exp}/run.sh"
    
    jobname=$(echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl && basename ${ssl_tokenizer_exp} || echo "${ssl_tokenizer_exp}/train.log")
    
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${ssl_tokenizer_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${ssl_tokenizer_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.beats_tokenizer_train \
            --use_preprocessor true \
            --beats_teacher_ckpt_path "${teacher_ckpt_path_}" \
            --train_data_path_and_name_and_type "${_ssl_train_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
            --train_shape_file "${ssl_stats_dir}/train/speech_shape" \
            --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
            --resume true \
            --fold_length "${speech_fold_length}" \
            --output_dir "${ssl_tokenizer_exp}" \
            ${_opts} ${beats_args}
    
    # Convert tokenizer checkpoint to float32 for inference
    log "Generating float32 checkpoint from tokenizer training"
    checkpoint_path="${ssl_tokenizer_exp}/epoch_latest.pt"
    generate_checkpoint "${ssl_tokenizer_exp}" "${checkpoint_path}"
}

tokenizer_inference() {
    iteration=$1
    ssl_tokenizer_exp="${expdir}/beats_tokenizer_iter${iteration}_${ssl_tag}"
    
    _opts=""
    if [ -n "${external_tokenizer_model}" ]; then
        _opts+="--checkpoint_path ${external_tokenizer_model} "
    else
        tokenizer_checkpoint_path_="${ssl_tokenizer_exp}/epoch_latest.pt"
        if [ ! -f "${tokenizer_checkpoint_path_}" ]; then
            log "Generating tokenizer checkpoint for inference"
            generate_checkpoint "${ssl_tokenizer_exp}" "${tokenizer_checkpoint_path_}"
        fi
        _opts+="--checkpoint_path ${tokenizer_checkpoint_path_} "
        _opts+="--config_path ${ssl_tokenizer_exp}/config.yaml "
    fi
    _opts+="${_waveform_opt} "
    
    _nj=$((ngpu==0?nj:ngpu))
    _ngpu=$((ngpu==0?0:1))
    
    for _data_dir in "${_ssl_valid_dir}" "${_ssl_train_dir}"; do
        ./scripts/feats/audio_tokenization.sh \
            --codec_choice beats \
            --file_name ${_scp} \
            --src_dir "${_data_dir}" \
            --tgt_dir "${_data_dir}/iter${iteration}_${_tokenizer_inference_tag}" \
            --nj "${_nj}" --ngpu "${_ngpu}" --batch_size "${tokenizer_inference_batch_size}" \
            ${_opts}

        cp "${_data_dir}/iter${iteration}_${_tokenizer_inference_tag}/${_scp%.scp}_beats.txt" \
            "${_data_dir}/target_iter${iteration}_${_tokenizer_inference_tag}"
    done
}

if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: BEATs Random Tokenization: ${data_feats}/${train_set}, ${data_feats}/${valid_set}"
        setup_common_vars
        
        _opts=
        if [ -n "${tokenizer_inference_config}" ]; then
            _opts+="--config_path ${tokenizer_inference_config} "
        fi
        _opts+="${_waveform_opt} "

        # Tokenize
        # TODO(shikhar): Undo changes to dump_codec.py
        _nj=$((ngpu==0?nj:ngpu))
        _ngpu=$((ngpu==0?0:1))
        for dset in "${valid_set}" "${train_set}"; do
            ./scripts/feats/audio_tokenization.sh \
                --codec_choice beats_random \
                --file_name ${_scp} \
                --src_dir "${data_feats}/${dset}" \
                --tgt_dir "${data_feats}/${dset}/iter0_${_tokenizer_inference_tag}" \
                ${_opts} \
                --nj "${_nj}" --ngpu ${_ngpu} --batch_size "${tokenizer_inference_batch_size}"
            cp "${data_feats}/${dset}/iter0_${_tokenizer_inference_tag}/${_scp%.scp}_beats_random.txt" "${data_feats}/${dset}/target_iter0_${_tokenizer_inference_tag}"
        done
        
        # Prepare token list
        : > "${token_listdir}/tokens.txt"  # Clear the file if it exists
        echo "<unk>" >> "${token_listdir}/tokens.txt"
        for i in $(seq 0 $((n_targets - 1))); do
            echo "${i}" >> "${token_listdir}/tokens.txt"
        done
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        setup_common_vars

        log "Stage 6: BEATs collect stats: train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            _opts+="--config ${train_config} "
        fi
        _opts+="${_waveform_opt} "
        
        # 1. Split the key file
        _logdir="${ssl_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_ssl_train_dir}/${_scp} wc -l)" "$(<${_ssl_valid_dir}/${_scp} wc -l)")

        key_file="${_ssl_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_ssl_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${ssl_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${ssl_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${ssl_stats_dir}/run.sh";chmod +x "${ssl_stats_dir}/run.sh"

        # 3. Submit jobs
        log "BEATs collect-stats started... log: '${_logdir}/stats.*.log'"
        
        # Run collectstats
        # shellcheck disableSC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.beats_train \
                --collect_stats true \
                --use_preprocessor true \
                --token_list "${token_listdir}/tokens.txt" \
                --train_data_path_and_name_and_type "${_ssl_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_ssl_train_dir}/target_iter0_${_tokenizer_inference_tag},target,text" \
                --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_ssl_valid_dir}/target_iter0_${_tokenizer_inference_tag},target,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${beats_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        _opts+="--skip_sum_stats"
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${ssl_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${ssl_stats_dir}/train/target_shape" \
            awk -v N="$(<${token_listdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
            >"${ssl_stats_dir}/train/target_shape.word"

        <"${ssl_stats_dir}/valid/target_shape" \
            awk -v N="$(<${token_listdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
            >"${ssl_stats_dir}/valid/target_shape.word"
    fi
    
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        setup_common_vars
        log "Stage 7: BEATs Training: train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"
        for ((iter=${train_start_iter}; iter<=${train_stop_iter};iter++)); do
            log "Starting iteration ${iter} of BEATs Training"
            if ! [ ${iter} -eq 0 ]; then
                if [ -z "${external_tokenizer_model}" ]; then
                    train_tokenizer ${iter}
                else
                    if [ "${train_start_iter}" -ne "${train_stop_iter}" ]; then
                        log "Error: External tokenizer model is provided, but training is requested for multiple iterations"
                        exit 1
                    fi
                fi
                tokenizer_inference ${iter}
            fi
            train_encoder ${iter}
        done
    fi
else
    log "Skip the training stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"