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
skip_packing=true    # Skip the packing stage.
skip_upload_hf=true  # Skip uploading to huggingface stage.
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
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw). 
                    # flac does not work with kaldi during audio tokenization phase
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Pretrain model related
ssl_tag=       # Suffix to the result dir for ssl model training.
beats_args=         # Arguments for ssl model training, e.g., "--max_epoch 10".
                     # Note that it will overwrite args in ssl config.
num_splits_ssl=1 # Number of splitting for lm corpus.

# Pretrain related
train_start_iter= # Pretrain starts from the specified iteration (0 mean MFCC iteraion)
train_stop_iter=  # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion)
train_config=    # Configration file of training stage
n_targets=             # Number of codebook targets

# Upload model related
hf_repo=
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
    --skip_packing   # Skip the packing stage (default="${skip_packing}").
    --skip_upload_hf # Skip uploading to huggingface stage (default="${skip_upload_hf}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --datadir        # Directory to save the prepared data from Stage 1 (default="${datadir}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --ssl_tag        # Suffix to the result dir for ssl model training (default="${ssl_tag}").
    --python         # Specify python to execute espnet commands (default="${python}").
    --hf_repo        # Hugging face repository name (default="${hf_repo}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
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
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi
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

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if ! [ "${feats_type}" = raw ]; then
            log "Error in Stage 2: not supported: --feats_type ${feats_type}"
            exit 2
        fi

        log "Stage 2: Format wav.scp: ${datadir}/ -> ${data_feats}"
        for dset in "${train_set}" "${valid_set}"; do
            utils/copy_data_dir.sh --validate_opts --non-print ${datadir}/"${dset}" "${data_feats}/org/${dset}"
            rm -f ${data_feats}/org/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                                            --audio-format "${audio_format}" --fs "${fs}" \
                                            "${datadir}/${dset}/wav.scp" "${data_feats}/org/${dset}"

            echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"
            # Copy data from multiple jobs
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
        done
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do

            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if ! [ "${_feats_type}" = raw ]; then
                log "Error: not supported: --feats_type ${feats_type}"
                exit 2
            fi

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
            awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
            '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
            >"${data_feats}/${dset}/utt2num_samples"
            
            <"${data_feats}/org/${dset}/wav.scp" \
            utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
            >"${data_feats}/${dset}/wav.scp"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the stages for data preparation"
fi


if ! "${skip_train}"; then
    codec_choice="beats_random"
    if ! [ ${train_start_iter} -eq 0 ]; then
        codec_choice="beats"
    fi
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: BEATs Tokenization: ${data_feats}/${train_set}, ${data_feats}/${valid_set}"
        
        # Tokenize
        _nj=$((ngpu==0?nj:ngpu))
        for dset in "${train_set}" "${valid_set}"; do
            ./scripts/feats/audio_tokenization.sh \
                --codec_choice ${codec_choice} \
                --file_name wav.scp \
                --src_dir "${data_feats}/${dset}" \
                --tgt_dir "${data_feats}/${dset}/codes" \
                --nj "${_nj}"
            cp "${data_feats}/${dset}/codes/wav_${codec_choice}.txt" "${data_feats}/${dset}/target_iter0"
        done
        
        # Prepare token list
        : > "${token_listdir}/tokens.txt"  # Clear the file if it exists
        echo "<unk>" >> "${token_listdir}/tokens.txt"
        for i in $(seq 0 $((n_targets - 1))); do
            echo "${i}" >> "${token_listdir}/tokens.txt"
        done
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _ssl_train_dir="${data_feats}/${train_set}"
        _ssl_valid_dir="${data_feats}/${valid_set}"

        log "Stage 5: BEATs collect stats: train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            _opts+="--config ${train_config} "
        fi

        _feats_type="$(<${_ssl_train_dir}/feats_type)"
        if ! [ "${_feats_type}" = raw ]; then
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi

        _scp=wav.scp
        _type=sound
        
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
        log "Generate '${ssl_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${ssl_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${ssl_stats_dir}/run.sh";chmod +x "${ssl_stats_dir}/run.sh"

        # 3. Submit jobs
        log "BEATs collect-stats started... log: '${_logdir}/stats.*.log'"
        
        # shellcheck disableSC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.beats_train \
                --collect_stats true \
                --use_preprocessor true \
                --token_list "${token_listdir}/tokens.txt" \
                --train_data_path_and_name_and_type "${_ssl_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_ssl_train_dir}/target_iter0,target,text" \
                --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_ssl_valid_dir}/target_iter0,target,text" \
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
    
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _ssl_train_dir="${data_feats}/${train_set}"
        _ssl_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: BEATs Training: train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"
        for ((iter=${train_start_iter}; iter<=${train_stop_iter};iter++)); do
            log "Iter ${iter} BEATs Training started..."
            
            if ! [ ${iter} -eq 0 ]; then
                # Train beats tokenizer and generate codes from it
                _checkpoint_path="TODODODO"
                _config_path=""
                log "Implement the training stage for iter ${iter}!"
                exit 2
                log "Iter ${iter} BEATs Tokenizer Training completed, model saved in: ${ssl_exp}"
                # Tokenize
                for _data_dir in "${_ssl_train_dir}" "${_ssl_val_dir}"; do
                    _nj=$((ngpu==0?nj:ngpu))
                    ./scripts/feats/audio_tokenization.sh \
                        --codec_choice beats \
                        --file_name wav.scp \
                        --src_dir "${_data_dir}" \
                        --tgt_dir "${_data_dir}/codes_iter${iter}" \
                        --checkpoint_path "${_checkpoint_path}" \
                        --config_path "${_config_path}" \
                        --nj "${_nj}"
                    cp "${_data_dir}/codes/wav_${codec_choice}.txt" "${_data_dir}/target_iter${iter}"
                done
            fi

            # Train beats encoder model
            ssl_exp="${expdir}/beats_iter${iter}_${ssl_tag}"

            _opts=
            if [ -n "${train_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.beats_train --print_config --optim adam
                _opts+="--config ${train_config} "
            fi

            _feats_type="$(<${_ssl_train_dir}/feats_type)"
            if ! [ "${_feats_type}" = raw ]; then
                log "Error: not supported: --feats_type ${feats_type}"
                exit 2
            fi
            _scp=wav.scp
            _type=sound

            if [ "${num_splits_ssl}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${ssl_stats_dir}/splits${num_splits_ssl}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                        --scps \
                        "${_ssl_train_dir}/${_scp}" \
                        "${_ssl_train_dir}/target_iter${iter}" \
                        "${ssl_stats_dir}/train/speech_shape" \
                        "${ssl_stats_dir}/train/target_shape.word" \
                        --num_splits "${num_splits_ssl}" \
                        --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/target_iter${iter},target,text "
                _opts+="--train_shape_file ${_split_dir}/speech_shape "
                _opts+="--train_shape_file ${_split_dir}/target_shape.word "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/target_iter${iter},target,text "
                _opts+="--train_shape_file ${ssl_stats_dir}/train/speech_shape "
                _opts+="--train_shape_file ${ssl_stats_dir}/train/target_shape.word "
            fi

            log "Generate '${ssl_exp}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${ssl_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${ssl_exp}/run.sh"; chmod +x "${ssl_exp}/run.sh"

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
            log "BEATs Training started... log: '${ssl_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${ssl_exp})"
            else
                jobname="${ssl_exp}/train.log"
            fi
            
            # shellcheck disable=SC2086
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
                    --valid_data_path_and_name_and_type "${_ssl_valid_dir}/target_iter${iter},target,text" \
                    --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
                    --valid_shape_file "${ssl_stats_dir}/valid/target_shape.word" \
                    --resume true \
                    --fold_length "${speech_fold_length}" \
                    --fold_length "${text_fold_length}" \
                    --output_dir "${ssl_exp}" \
                    ${_opts} ${beats_args}

            log "Iter ${iter} BEATs Training completed, model saved in: ${ssl_exp}"
        done
    fi
else
    log "Skip the training stages"
fi



# packed_model="${ssl_exp}/${ssl_exp##*/}_${inference_ssl_model%.*}.zip"
# Skip pack preparation if using a downloaded model
# if ! "${skip_packing}"; then
#     if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
#         log "Stage 8: Pack model: ${packed_model}"

#         _opts=
#         if [ "${feats_normalize}" = global_mvn ]; then
#             _opts+="--option ${ssl_stats_dir}/train/feats_stats.npz "
#         fi
#         # shellcheck disable=SC2086
#         ${python} -m espnet2.bin.pack ssl \
#             --ssl_train_config "${ssl_exp}"/config.yaml \
#             --ssl_model_file "${ssl_exp}"/"${inference_ssl_model}" \
#             ${_opts} \
#             --option "${ssl_exp}"/images \
#             --option "${expdir}/${km_tag}/km_${n_targets_list[${train_stop_iter}]}.mdl" \
#             --outpath "${packed_model}"
#     fi
# else
#     log "Skip the packing stage"
# fi

# if ! "${skip_upload_hf}"; then
#     if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
#         [ -z "${hf_repo}" ] && \
#             log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
#         exit 1
#         log "Stage 9: Upload model to HuggingFace: ${hf_repo}"

#         if [ ! -f "${packed_model}" ]; then
#             log "ERROR: ${packed_model} does not exist. Please run stage 8 first."
#             exit 1
#         fi

#         gitlfs=$(git lfs --version 2> /dev/null || true)
#         [ -z "${gitlfs}" ] && \
#             log "ERROR: You need to install git-lfs first" && \
#             exit 1

#         dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
#         [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

#         if command -v git &> /dev/null; then
#             _creator_name="$(git config user.name)"
#             _checkout="git checkout $(git show -s --format=%H)"
#         else
#             _creator_name="$(whoami)"
#             _checkout=""
#         fi
#         # /some/where/espnet/egs2/foo/ssl1/ -> foo/ssl1
#         _task="$(pwd | rev | cut -d/ -f2 | rev)"
#         # foo/ssl1 -> foo
#         _corpus="${_task%/*}"
#         _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

#         # copy files in ${dir_repo}
#         unzip -o ${packed_model} -d ${dir_repo}
#         # Generate description file
#         # shellcheck disable=SC2034
#         hf_task=self-supervised-learning
#         # shellcheck disable=SC2034
#         espnet_task=SSL
#         # shellcheck disable=SC2034
#         task_exp=${ssl_exp}
#         eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

#         this_folder=${PWD}
#         cd ${dir_repo}
#         if [ -n "$(git status --porcelain)" ]; then
#             git add .
#             git commit -m "Update model"
#         fi
#         git push
#         cd ${this_folder}
#     fi
# else
#     log "Skip the uploading to HuggingFace stage"
# fi

log "Successfully finished. [elapsed=${SECONDS}s]"
