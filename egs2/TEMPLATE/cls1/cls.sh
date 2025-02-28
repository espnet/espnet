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
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip uploading to huggingface
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
datadir=data         # Directory to save data from Stage 1.
dumpdir=dump         # Directory to dump features.
inference_nj=4      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands
use_lightning=false     # Whether to use pytorch lightning trainer for training.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark.
fs=16k               # Sampling rate.
speech_fold_length=160000 # The length of the speech input to the cls model.
label_fold_length=1   # fold_length for labels during CLS training. Set to 1 for multi-class classification.
cls_stats_dir=      # The directory used for collect-stats mode.

# data preprpocessing related
min_wav_duration=0.1 # Minimum duration in seconds to use in training
max_wav_duration=  # Maximum duration in seconds to use in training

# cls model related
cls_tag=    # Suffix to the result dir for cls model training.
cls_config= # Config for cls model training.
cls_args=   # Arguments for cls model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in cls config.
feats_normalize=uttmvn # Normalizaton layer type.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
classification_type=        # Type of classification task, multi-class or multi-label

# cls inference related
download_model=
inference_model=valid.acc.best.pth
inference_tag=    # Suffix to the inference dir for cls model inference
output_all_probabilities=true

hf_repo=        # Huggingface repo name

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.

text_classes=   # Path to the text file containing all classes.

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
    --datadir        # Directory to save data from Stage 1 (default="${datadir}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").
    --use_lightning # Whether to use pytorch lightning trainer for training (default="${use_lightning}").
    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").
    # Feature extraction related
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --speech_fold_length # The length of the speech input to the cls model (default="${speech_fold_length}").
    --label_fold_length  # fold_length for labels during CLS training (default="${label_fold_length}").
    --cls_stats_dir   # The directory used for collect-stats mode (default="${cls_stats_dir}").
    --min_wav_duration   # Minimum duration in seconds to use in training (default="${min_wav_duration}").
    --max_wav_duration   # Maximum duration in seconds to use in training (default="${max_wav_duration}").
    # cls model related
    --cls_tag        # Suffix to the result dir for classification model training (default="${cls_tag}").
    --cls_config     # Config for classification model training (default="${cls_config}").
    --cls_args       # Arguments for classification model training, e.g., "--max_epoch 10" (default="${cls_args}").
                      # Note that it will overwrite args in cls config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --pretrained_model # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch # Ignore initial mismatch (default="${ignore_init_mismatch}").
    --classification_type # Type of classification task, multi-class or multi-label (default="${classification_type}").
    # cls inference related
    --download_model  # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --inference_model  # classification model path for inference (default="${inference_model}").
    --inference_tag    # Suffix to the inference dir for cls model inference
    --output_all_probabilities # Output all probabilities in the inference stage (default="${output_all_probabilities}").
    # Huggingface related
    --hf_repo        # Huggingface repo name (default="${hf_repo}").
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set               # Name of training set (required).
    --valid_set               # Name of development set (required).
    --test_sets               # Names of evaluation sets (required).
    --text_classes            # Path to the text file containing classes (default="${text_classes}").
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


data_feats=${dumpdir}

# Set tag for naming of model directory
if [ -z "${cls_tag}" ]; then
    if [ -n "${cls_config}" ]; then
        cls_tag="$(basename "${cls_config}" .yaml)"
    else
        cls_tag="train"
    fi
    if [ -n "${cls_args}" ]; then
        cls_tag+="$(echo "${cls_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

if [ -z "${inference_tag}" ]; then
    inference_tag=inference
fi

[ -z "${text_classes}" ] && text_classes="${data_feats}/${train_set}/text"

# The directory used for collect-stats mode
if [ -z "${cls_stats_dir}" ]; then
    cls_stats_dir="${expdir}/cls_stats_${fs}"
fi
# The directory used for training commands
cls_exp="${expdir}/cls_${cls_tag}"
token_list=${datadir}/token_list


if [[ "${classification_type}" == "multi-label" ]]; then
    if [[ "${use_lightning}" != "true" ]]; then
        log "Multi-label classification is only supported with PyTorch Lightning trainer. Please set --use_lightning true."
        exit 1
    fi
fi
# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for ${datadir}/${train_set}, ${datadir}/${valid_set}, etc."
        local/data.sh ${datadir} ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Stage 2: Format wav.scp: ${datadir}/ -> ${data_feats}"
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path
        # with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            utils/copy_data_dir.sh --validate_opts --non-print "${datadir}/${dset}" "${data_feats}/${dset}"
            rm -f ${data_feats}/${dset}/{wav.scp,reco2file_and_channel}

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}"  \
                "${datadir}/${dset}/wav.scp" "${data_feats}/${dset}"

            # Note(jiatong): default use raw as feats_type, see more types in other TEMPLATE recipes
            echo "raw" > "${data_feats}/${dset}/feats_type"
        done
    fi

    # if max_wav_duration is not set skip stage 3
    if [ -z "${max_wav_duration}" ]; then
        log "max_wav_duration is not set, skipping stage 3 (Modify long/short data)"
        if [ ${stage} -le 3 ]; then
            stage=4
        fi
    fi
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Modify long/short data"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)" #must be raw
            if [ "${_feats_type}" != raw ]; then
                log "feats_type is not raw: ${_feats_type}"
                exit 1
            fi

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")
            # keep copy of original utt2num_samples and wav.scp
            cp "${data_feats}/${dset}/utt2num_samples" "${data_feats}/${dset}/utt2num_samples.org"
            cp "${data_feats}/${dset}/wav.scp" "${data_feats}/${dset}/wav.org.scp"

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/${dset}/utt2num_samples.org" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/${dset}/wav.org.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"
            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
            # Report the change in the number of utterances
            _org_num_utts=$(wc -l "${data_feats}/${dset}/wav.org.scp" | awk '{print $1}')
            _new_num_utts=$(wc -l "${data_feats}/${dset}/wav.scp" | awk '{print $1}')
            log "Filtered ${dset}: ${_new_num_utts} remain out of ${_org_num_utts}!"
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Generate token_list covering all classes from ${text_classes}"
        ${python} -m espnet2.bin.tokenize_text --token_type "word" \
            --input "${text_classes}" --output "${token_list}" \
            --field 2- --write_vocabulary true --add_symbol "<unk>:-1"
            # unk is just a dummy symbol for compatibility,
            # we ensure that it is not used in the cls model
    fi
else
    log "Skip the data preparation stages"
fi


if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _cls_train_dir="${data_feats}/${train_set}"
        _cls_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: cls collect stats: train_set=${_cls_train_dir}, valid_set=${_cls_valid_dir}"

        _opts=
        if [ -n "${cls_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.cls_train --print_config --optim adam
            _opts+="--config ${cls_config} "
        fi

        _feats_type="$(<${_cls_train_dir}/feats_type)"
        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _opts+="--frontend_conf fs=${fs} "

        # 1. Split the key file
        _logdir="${cls_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_cls_train_dir}/${_scp} wc -l)" "$(<${_cls_valid_dir}/${_scp} wc -l)")

        key_file="${_cls_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_cls_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${cls_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${cls_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${cls_stats_dir}/run.sh"; chmod +x "${cls_stats_dir}/run.sh"

        # 3. Submit jobs
        log "cls collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.cls_train \
                --collect_stats true \
                --token_type "word" \
                --use_preprocessor true \
                --train_data_path_and_name_and_type "${_cls_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_cls_train_dir}/text,label,text" \
                --valid_data_path_and_name_and_type "${_cls_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_cls_valid_dir}/text,label,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                --token_list "${token_list}" \
                ${_opts} ${cls_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${cls_stats_dir}"

    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _cls_train_dir="${data_feats}/${train_set}"
        _cls_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: cls Training: train_set=${_cls_train_dir}, valid_set=${_cls_valid_dir}"

        _opts=
        if [ -n "${cls_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.cls_train --print_config --optim adam
            _opts+="--config ${cls_config} "
        fi

        _feats_type="$(<${_cls_train_dir}/feats_type)"
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            _type=sound
        fi
        _opts+="--frontend_conf fs=${fs} "

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${cls_stats_dir}/train/feats_stats.npz "
        fi

        _opts+="--train_data_path_and_name_and_type ${_cls_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_cls_train_dir}/text,label,text "
        _opts+="--train_shape_file ${cls_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${cls_stats_dir}/train/label_shape "

        _opts+="--valid_data_path_and_name_and_type ${_cls_valid_dir}/${_scp},speech,${_type} "
        _opts+="--valid_data_path_and_name_and_type ${_cls_valid_dir}/text,label,text "
        _opts+="--valid_shape_file ${cls_stats_dir}/valid/speech_shape "
        _opts+="--valid_shape_file ${cls_stats_dir}/valid/label_shape "
        _opts+="--token_list ${token_list} "
        _opts+="--token_type word "
        _opts+="--fold_length ${speech_fold_length} "
        _opts+="--fold_length ${label_fold_length} "

        log "Generate '${cls_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${cls_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${cls_exp}/run.sh"; chmod +x "${cls_exp}/run.sh"

        log "cls training started... log: '${cls_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${cls_exp})"
        else
            jobname="${cls_exp}/train.log"
        fi

        if "${use_lightning}"; then
            log "Use PyTorch Lightning trainer"
            ${python} pyscripts/utils/rotate_logfile.py "${cls_exp}"/train.log

            ${cuda_cmd} --name "${jobname}" \
                --gpu "${ngpu}" \
                --num_tasks "${ngpu}" \
                --num_nodes "${num_nodes}" \
                "${cls_exp}"/train.log \
                srun --export=ALL \
                ${python} -m espnet2.bin.lightning_train \
                    --task cls \
                    --lightning_conf "{devices: ${ngpu}, num_nodes: ${num_nodes}, default_root_dir: ${cls_exp}}" \
                    --user_callbacks mAP_logging \
                    --use_preprocessor true \
                    --resume true \
                    ${pretrained_model:+--init_param $pretrained_model} \
                    --ignore_init_mismatch ${ignore_init_mismatch} \
                    --output_dir "${cls_exp}" \
                    ${_opts} ${cls_args}
        else
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${cls_exp}/train.log" \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${cls_exp}/.dist_init_" \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.cls_train \
                    --use_preprocessor true \
                    ${pretrained_model:+--init_param $pretrained_model} \
                    --resume true \
                    --output_dir "${cls_exp}" \
                    ${_opts} ${cls_args}
        fi
    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Using ${download_model} for decoding and evaluation"
    cls_exp="${expdir}/$(basename "${download_model}")"
    mkdir -p "${cls_exp}"

    # Ensure huggingface-cli is installed
    command -v huggingface-cli &>/dev/null || log "huggingface-cli not found, please install it"

    huggingface-cli download "${download_model}" --local-dir ./
    mv ./**/exp/*/* "${cls_exp}/" && log "Files moved." || log "Warning: An error occurred in moving. Please recheck the downloaded huggingface model."

    # Set variables for stages below
    inference_model=$(echo ${cls_exp}/*epoch.pth)
    inference_model=$(basename "$inference_model")
fi
if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Predict with models: training_dir=${cls_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${cls_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${cls_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${cls_exp}/run.sh"; chmod +x "${cls_exp}/run.sh"
        _opts=

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${cls_exp}/cls_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            _type=sound
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit inference jobs
            log "cls inference started... log: '${_logdir}/cls_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/cls_inference.JOB.log \
                ${python} -m espnet2.bin.cls_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --classification_train_config "${cls_exp}/config.yaml" \
                    --classification_model_file "${cls_exp}/${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --output_all_probabilities ${output_all_probabilities} \
                    ${_opts} || { cat $(grep -l -i error "${_logdir}"/cls_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/prediction/${f}"
                done | LC_ALL=C sort -k1 > "${_dir}/${f}"
            done
        done
    fi

    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Scoring"
        _cmd=${decode_cmd}

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _inf_dir="${cls_exp}/cls_${dset}"
            _dir="${cls_exp}/cls_${dset}/scoring"
            mkdir -p "${_dir}"
            python3 pyscripts/utils/cls_score.py \
                -gtxt "${_data}/text" \
                -ptxt "${_inf_dir}/text" \
                -pscore "${_inf_dir}/score" \
                -tok "${token_list}" > "${_dir}"/metrics
        done

        scripts/utils/show_cls_result.sh "${cls_exp}" > "${cls_exp}"/RESULTS.md
        cat "${cls_exp}"/RESULTS.md
    fi
else
    log "Skip the evaluation stages"
fi

if ! "${skip_upload}"; then
    packed_model="${cls_exp}/${cls_exp##*/}_${inference_model%.*}.zip"
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Pack models for uploading to huggingface: ${cls_exp}"
        _opts=
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${cls_stats_dir}/train/feats_stats.npz "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack cls \
            --classification_train_config "${cls_exp}"/config.yaml \
            --classification_model_file "${cls_exp}"/"${inference_model}" \
            ${_opts} \
            --option "${cls_exp}"/RESULTS.md \
            --option "${cls_exp}"/images \
            --option "${token_list}" \
            --outpath "${packed_model}"
    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: Upload to huggingface"
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
        exit 1
        if [ ! -f "${packed_model}" ]; then
            log "ERROR: ${packed_model} does not exist. Please run previous stage first."
            exit 1
        fi

        gitlfs=$(git lfs --version 2> /dev/null || true)
        [ -z "${gitlfs}" ] && \
            log "ERROR: You need to install git-lfs first" && \
            exit 1

        dir_repo=${expdir}/hf_${hf_repo//"/"/"_"}
        [ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="git checkout $(git show -s --format=%H)"
        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/cls1/ -> foo/cls1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/cls1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=classification
        # shellcheck disable=SC2034
        espnet_task=CLS
        lang=en
        # shellcheck disable=SC2034
        task_exp=${cls_exp}
        eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

        this_folder=${PWD}
        cd ${dir_repo}
        if [ -n "$(git status --porcelain)" ]; then
            git add .
            git commit -m "Update model"
        fi
        git push
        cd ${this_folder}
    fi
else
    log "Skip the upload stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
