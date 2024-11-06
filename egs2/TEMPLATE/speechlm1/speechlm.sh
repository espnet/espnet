#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts=""  # Options to be passed to local/data.sh.
data_tag=""         # You may combine the data in multiple ways. This is the tag for data composition

# Audio Feature extraction related
feats_type=raw             # Input feature type.
audio_format=flac          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=30        # Maximum duration in second.
fs=16000                   # Sampling rate.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
speechlm_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
speechlm_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for tts corpus.

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=valid.acc.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file.
download_model=""  # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
task=            # when task is multi_task, skip data preparation and use train/valid yamls
train_jsons=""
valid_jsons=""
test_jsons=""

# Tokenization related
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.
tokenization_choices=""
codec_choice="EnCodec"
codec_checkpoint_path=null
codec_config_path=null
semantic_choice="WavLM"
semantic_opts=""
g2p="g2p_en"
nlsyms_txt=none
cleaner=tacotron
token_list_dir=

# TODO(Jinchuan): Upload model related
hf_repo=

help_message=""

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

if [ -z ${task} ]; then
    echo "Task is not specified" && exit 1;
fi

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw_${task}"
    data_audio="${dumpdir}/audio_raw"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)"
    else
        echo "No training configuration found ..." && exit 1;
    fi

    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

if [ -z ${data_tag} ]; then
    data_tag=${train_set}_${task}
fi

# The directory used for collect-stats mode
if [ -z "${speechlm_stats_dir}" ]; then
    speechlm_stats_dir="${expdir}/speechlm_stats_${data_tag}"
fi
# The directory used for training commands
if [ -z "${speechlm_exp}" ]; then
    speechlm_exp="${expdir}/speechlm_${tag}"
fi

if [ -z ${token_list_dir} ]; then
    token_list_dir=data/token_list/${data_tag}
fi

if [ ${task} == "multi_task" ]; then
    if [ -z ${train_yamls} ] || [ -z ${valid_yamls} ] || !${skip_data_prep}; then
        echo "When training with multi-task, we skip data preparation stages"
        echo "and directly use the prepared train/valid yaml files"
        echo "You can prepare the yaml files for each task one by one"
        echo "and merge them as: "
        echo "speechlm.sh --train_yamls 'a.yaml,b.yaml' --valid_yamls 'a.yaml,b.yaml'" || exit 1;
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        # TODO(jinchuan): only consider the wav.scp. In fact we could consider multiple
        # audio scp files. Same as in stage 3

        log "Stage 2: Format wav.scp: data/ -> ${data_audio}/"
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_audio}${_suf}/${dset}"
            rm -f ${data_audio}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_audio}${_suf}/${dset}"
            echo "${feats_type}" > "${data_audio}${_suf}/${dset}/feats_type"
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_audio}/org -> ${data_audio}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_audio}/org/${dset}" "${data_audio}/${dset}"
            cp "${data_audio}/org/${dset}/feats_type" "${data_audio}/${dset}/feats_type"

            # Remove short utterances
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_audio}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_audio}/${dset}/utt2num_samples"
            <"${data_audio}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_audio}/${dset}/utt2num_samples"  \
                >"${data_audio}/${dset}/wav.scp"

            # Remove empty text
            <"${data_audio}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_audio}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            # shellcheck disable=SC2086
            utils/fix_data_dir.sh "${data_audio}/${dset}"
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        # Skip at this moment. Mainly for SSL K-Means
        log "Stage 4: Train necessary models before tokenization"

    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Prepare each data entry for the given task"
        if ${skip_train}; then
            _dsets=${test_sets}
        else
            _dsets="${train_set} ${valid_set} ${test_sets}"
        fi

        # Parse the data preparation operations from Python task definition.
        prepare_opts=$(python -c "from espnet2.speechlm.definitions import tasks; print(tasks['${task}'].find_modality_type)")

        for dset in ${_dsets}; do
            opts=""
            for prepare_opt in ${prepare_opts}; do
                mkdir -p ${data_feats}/${dset}/token_lists

                IFS=',' read -r _name _modality _type <<< "${prepare_opt}"
                # for discrete operations, we will also generate a vocabulary.

                if [ ! -f ${data_audio}/${dset}/${_name} ]; then
                    echo "File ${data_audio}/${dset}/${_name} is missing. Exit" || exit 1;
                fi

                if [ ${_modality} == "ssl" ]; then
                    echo "do ssl tokenization" && exit 1;

                elif [ ${_modality} == "codec" ]; then
                    echo "Codec Tokenization: ${data_audio}/${dset}/${_name} -> ${data_feats}/${dset}/${_name}"
                    scripts/feats/codec_tokenization.sh \
                        --src_dir ${data_audio}/${dset} --tgt_dir ${data_feats}/${dset} \
                        --codec_fs ${fs} --dump_audio false \
                        --file_name ${_name} --nj ${nj} --codec_choice ${codec_choice} \
                        --checkpoint_path ${codec_checkpoint_path} \
                        --config_path ${codec_config_path}

                elif [ ${_modality} == "g2p" ]; then
                    echo "Find G2P vocabulary and copy text"
                    ${python} -m espnet2.bin.tokenize_text \
                        --token_type "phn" -f 2- \
                        --input "${data_audio}/${dset}/${_name}" \
                        --output "${data_feats}/${dset}/token_lists/g2p_token_list" \
                        --non_linguistic_symbols "${nlsyms_txt}" \
                        --cleaner "${cleaner}" \
                        --g2p "${g2p}" \
                        --write_vocabulary true
                    cp "${data_audio}/${dset}/${_name}" "${data_feats}/${dset}/${_name}"

                elif [ ${_modality} == "spk" ]; then
                    echo "copy utt2spk file"
                    cp "${data_audio}/${dset}/${_name}" "${data_feats}/${dset}/${_name}"

                else
                    echo "Unsupported modality ${_modality}" && exit 1;
                fi

                opts+="--file_modality_type ${data_feats}/${dset}/${_name},${_modality},${_type} "
                if [ -f ${data_feats}/${dset}/token_lists/${_modality}_token_list ]; then
                    opts+="--token_list ${data_feats}/${dset}/token_lists/${_modality}_token_list "
                fi
            done

            # The metadata for this dataset/task is saved in a yaml file
            ${python} pyscripts/utils/make_speechlm_json.py \
                --task ${task} \
                --output_json ${data_feats}/${dset}/data.json \
                ${opts}
        done
    fi

else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then

    _data_opts=""
    if [ -z ${train_jsons} ]; then
        train_jsons=${data_feats}/${train_set}/data.json
        log "No train_jsons provided. Use the prepared one: ${train_jsons}"
    fi

    for train_json in $train_jsons; do
        _data_opts+="--train_data_path_and_name_and_type ${train_json},_,dataset_json "
    done

    if [ -z ${valid_jsons} ]; then
        valid_jsons=${data_feats}/${valid_set}/data.json
        log "No valid_jsons provided. Use the prepared one: ${valid_jsons}"
    fi
    for valid_json in $valid_jsons; do
        _data_opts+="--valid_data_path_and_name_and_type ${valid_json},_,dataset_json "
    done

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Generate vocabulary from the given train_jsons"
        mkdir -p ${token_list_dir}
        ${python} pyscripts/utils/make_token_list_speechlm.py \
            --data_json ${train_jsons} \
            --token_list_dir ${token_list_dir}
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: SpeechLM collect stats: train_jsons=${train_jsons}, valid_set=${valid_jsons}"
        mkdir -p ${speechlm_stats_dir}

        _opts=
        if [ -n "${train_config}" ]; then
            _opts+="--config ${train_config} "
        fi

        # Since the training is based on multiple distinctive json files, we need to find a combined
        # example_list
        mkdir -p ${speechlm_stats_dir}/train
        ${python} pyscripts/utils/make_example_list_speechlm.py \
            --json_files ${train_jsons} \
            --output_file ${speechlm_stats_dir}/train/example_list

        mkdir -p ${speechlm_stats_dir}/valid
        ${python} pyscripts/utils/make_example_list_speechlm.py \
            --json_files ${valid_jsons} \
            --output_file ${speechlm_stats_dir}/valid/example_list

        # 1. Split the key file
        _logdir="${speechlm_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(wc -l ${speechlm_stats_dir}/train/example_list)" "$(wc -l ${speechlm_stats_dir}/valid/example_list)")

        key_file="${speechlm_stats_dir}/train/example_list"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train_sample_list.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${speechlm_stats_dir}/valid/example_list"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid_sample_list.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${speechlm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${speechlm_stats_dir}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${speechlm_stats_dir}/run.sh"; chmod +x "${speechlm_stats_dir}/run.sh"

        # 3. Submit jobs
        log "SpeechLM collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.speechlm_train" \
                --collect_stats true \
                --use_preprocessor true \
                --token_list ${token_list_dir}/token_list \
                --token_bias ${token_list_dir}/token_bias.json \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --multi_task_dataset true \
                --train_shape_file "${_logdir}/train_sample_list.JOB.scp" \
                --valid_shape_file "${_logdir}/valid_sample_list.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${_data_opts} ${train_args} \
                || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        _opts+="--skip_sum_stats"
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${speechlm_stats_dir}"

        # (Jinchuan) we only care about the #frames / #patches.
        for module in enc dec; do
            for dset in train valid; do
                if [ -f ${speechlm_stats_dir}/${dset}/${module}_seq_shape ]; then
                    cat ${speechlm_stats_dir}/${dset}/${module}_seq_shape |\
                    awk -F ',' '{print $1}' \
                    > ${speechlm_stats_dir}/${dset}/${module}_seq_lengths
                fi
            done
        done
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 8: SpeechlLM training: train_jsons=${train_jsons}, valid_set=${valid_jsons}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _data_opts+="--train_shape_file ${speechlm_stats_dir}/train/dec_seq_lengths "
        _data_opts+="--valid_shape_file ${speechlm_stats_dir}/valid/dec_seq_lengths "
        if [ -f ${speechlm_stats_dir}/train/enc_seq_shape ]; then
            _data_opts+="--train_shape_file ${speechlm_stats_dir}/train/enc_seq_lengths "
        fi
        if [ -f ${speechlm_stats_dir}/valid/enc_seq_shape ]; then
            _data_opts+="--valid_shape_file ${speechlm_stats_dir}/valid/enc_seq_lengths "
        fi

        log "Generate '${speechlm_exp}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${speechlm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${speechlm_exp}/run.sh"; chmod +x "${speechlm_exp}/run.sh"

        log "SpeechLM training started... log: '${speechlm_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${speechlm_exp})"
        else
            jobname="${speechlm_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${speechlm_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${speechlm_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.speechlm_train" \
                --use_preprocessor true \
                --token_list ${token_list_dir}/token_list \
                --token_bias ${token_list_dir}/token_bias.json \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --multi_task_dataset true \
                --resume true \
                --output_dir "${speechlm_exp}" \
                ${_opts} ${_data_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi

if [ -n "${download_model}" ]; then
    log "Use ${download_model} for inference and evaluation"
    # Not supported yet
fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Inference: training_dir=${speechlm_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        for dset in ${test_sets}; do
            test_json="${data_feats}/${dset}/data.json"
            _dir="${speechlm_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/log"
            mkdir -p ${_logdir}

            ${python} pyscripts/utils/make_example_list_speechlm.py \
                --json_files ${test_json} \
                --output_file ${_logdir}/example_list

            # 1. Split the key file
            key_file=${_logdir}/example_list
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/example_list.${n}"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/speechlm_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/speechlm_inference.JOB.log \
                ${python} -m espnet2.bin.speechlm_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type ${test_json},_,dataset_json \
                    --key_file "${_logdir}"/example_list.JOB \
                    --model_file "${speechlm_exp}"/"${inference_model}" \
                    --train_config "${speechlm_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} \
                    || { cat $(grep -l -i error "${_logdir}"/speechlm_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            # TODO
        done
    fi
else
    log "Skip the evaluation stages"
fi

# TODO(Jinchuan) Evaluation and model upload stages


log "Successfully finished. [elapsed=${SECONDS}s]"
