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
stop_stage=10     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
skip_upload_hf_data=true  # Skip uploading the prepared dataset to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related (stage 1-5)
local_data_opts=""  # Options to be passed to local/data.sh.
data_name=""        # The name of current dataset to prepare
task=               # when task is multi_task, skip data preparation and use train/valid jsons

# Data combination related (stage 6+)
data_combo_name=""  # The name of data combination for training, This usually means a combination of several
                    # datasets. Will only be used after data preparation stage.
train_jsons=""      # train/valid/test data json files that are prepared in advance and save locally
valid_jsons=""
test_jsons=""

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
nbest=1            # number of best hypotheses to generate during inference.

# Scoring related
scoring_args=
additional_ref_files=

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

# Tokenization related
# (1) codec
codec_choice="EnCodec" # codec
codec_checkpoint_path=null
codec_config_path=null
codec_hf_model_tag=null
codec_batch_size=3

# (2) ssl
ssl_choice="espnet_hubert" # currently only espnet_hubert
ssl_checkpoint_path=null
ssl_kmeans_path=null
ssl_nlayer=16
ssl_hf_model_tag=null
ssl_batch_bins=4800000

# (3) g2p
g2p="g2p_en"
cleaner=tacotron

# (4) text bpe
subword_choice=sentencepiece      # sentencepiece or huggingface
subword_model=                    # external subword model path or huggingface model tag
sentencepiece_choice=bpe          # bpe, unigram etc.
subword_train_text=               # text used to train subword model, if given
nbpe=5000
bpe_nlsyms=
bpe_input_sentence_size=10000000 # Size of input sentence for sentencepiece.
bpe_char_cover=1.0  # character coverage when modeling with sentencepiece.

# (5) Text LM embeddings
textlm_hf_model_tag=
textlm_max_words=1000

# (100) other general
nlsyms_txt=none
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

# Check for stage 1-5: data prep
if ! "${skip_data_prep}"; then
    if [ -z "${task}" ]; then
        log "Task is not specified but you want to prepare data." && exit 1;
    fi

    if [ -z ${data_name} ]; then
        log "Data_name is not specified but you want to prepare data." && exit 1;
    fi

        # Check feature type
    if [ "${feats_type}" = raw ]; then
        data_feats="${dumpdir}/raw_${task}_${data_name}"
        data_audio="${dumpdir}/audio_raw_${task}_${data_name}"
    else
        log "${help_message}"
        log "Error: only supported: --feats_type raw"
        exit 1;
    fi
fi
if ! ${skip_upload_hf_data} && ${skip_data_prep}; then
    echo "Should not skip data prep if you intend to upload data to HF." && exit 1;
fi

# Check for stage 6-7: data combination
if [ -n "${train_jsons}" ] && [ -n "${valid_jsons}" ]; then
    if [ -z "${data_combo_name}" ]; then
        log "External data resources are used. Please specify data_combo_name." && exit 1;
    fi
else
    if ! "${skip_data_prep}"; then
        if [ -z "${data_combo_name}" ]; then
            data_combo_name=${task}_${data_name}
        fi
    else
        log "No data from external resources or prepared locally. Cannot proceed..." && exit 1;
    fi
fi

if [ -z "${speechlm_stats_dir}" ]; then
    speechlm_stats_dir="${expdir}/speechlm_stats_${data_combo_name}"
fi

if [ "${subword_choice}" == "sentencepiece" ]; then
    if  [ ! -z "${subword_model}" ]; then
        if [ ! -f "${subword_model}".model ]; then
            log "subword_model is specified but not exist ... " && exit 1;
        fi
    else
        if ! "${skip_data_prep}"; then
            subword_model=${data_feats}/${train_set}/token_lists/text_bpe
        fi
    fi
else
    if [ -z "${subword_model}" ]; then
        log "To use HF tokenizer, you should specify the subword_model by the model tag" && exit 1;
    fi
fi

if [ -z ${token_list_dir} ]; then
    token_list_dir=data/token_list/${data_combo_name}
fi

# check for stage 8-9: training and inference
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="${data_combo_name}_$(basename "${train_config}" .yaml)"
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

if [ -z "${speechlm_exp}" ]; then
    speechlm_exp="${expdir}/speechlm_${tag}"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # NOTE(Jinchuan): We don't have to verify the data folders (like fix_data_dir.sh) here.
        # This will be done internally by pyscripts/utils/make_speechlm_json.py in stage 5.
        log "Stage 2: Format all audio files"

        all_triplets=$(python -c "from espnet2.speechlm.definitions import SPEECHLM_TASKS; print(SPEECHLM_TASKS['${task}'].data_triplets_string)")
        _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
        _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
        _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

        if ${skip_train}; then
            _dsets=${test_sets}
        else
            _dsets="${train_set} ${valid_set} ${test_sets}"
        fi

        for dset in ${_dsets}; do
            mkdir -p ${data_audio}/${dset}

            for triplet in ${all_triplets}; do
                IFS=',' read -r _name _modality _type <<< "${triplet}"

                if [ ${_modality} == "codec" ] || [ ${_modality} == "ssl" ] || [ ${_modality} == "codec_ssl" ]; then

                    # Format audio
                    _opts=
                    if [ -e data/"${dset}"/segments ]; then
                        _opts+="--segments data/${dset}/segments "
                    fi
                    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" \
                    --out_filename ${_name} ${_opts} \
                    "data/${dset}/${_name}" "${data_audio}/${dset}"

                    # Filter Length
                    if [[ ! " ${dset} " =~ " ${test_sets} " ]]; then
                        awk -v min_len="${_min_length}" -v max_len="${_max_length}" '
                        FNR==NR { lengths[$1]=$2; next }
                        ($1 in lengths) && (lengths[$1] >= min_len) && (lengths[$1] <= max_len) { print $0 }
                        ' ${data_audio}/${dset}/utt2num_samples ${data_audio}/${dset}/${_name} \
                        > ${data_audio}/${dset}/${_name}.tmp
                        mv ${data_audio}/${dset}/${_name}.tmp ${data_audio}/${dset}/${_name}
                    fi

                else
                    # Other non-speech items
                    <"data/${dset}/${_name}" \
                    awk ' { if( NF != 1 ) print $0; } ' >"${data_audio}/${dset}/${_name}"
                fi
            done
        done
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 4 ]; then
        log "Skip stage 3-4: No operations"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Prepare each data entry for the given task"
        if ${skip_train}; then
            _dsets=${test_sets}
        else
            _dsets="${train_set} ${valid_set} ${test_sets}"
        fi

        # Parse the data preparation operations from Python task definition.
        all_triplets=$(python -c "from espnet2.speechlm.definitions import SPEECHLM_TASKS; print(SPEECHLM_TASKS['${task}'].data_triplets_string)")

        for dset in ${_dsets}; do
            opts=""
            for triplet in ${all_triplets}; do
                mkdir -p ${data_feats}/${dset}/token_lists

                IFS=',' read -r _name _modality _type <<< "${triplet}"
                # for discrete operations, we will also generate a vocabulary.

                if [ ! -f ${data_audio}/${dset}/${_name} ]; then
                    log "File ${data_audio}/${dset}/${_name} is missing. Exit" && exit 1;
                fi

                if [ ${_modality} == "text_emb" ]; then
                    log "Offline Text LM inference for text embeddings"
                    scripts/feats/dump_textlm.sh \
                      --src_dir ${data_audio}/${dset} \
                      --tgt_dir ${data_feats}/${dset} \
                      --file_name ${_name} \
                      --hf_model_tag ${textlm_hf_model_tag} \
                      --max_words ${textlm_max_words} \
                      --nj ${nj}

                elif [ ${_modality} == "codec_ssl" ]; then
                    # do both codec and SSL tokenization and then splice them in time-axis
                    log "codec_ssl tokenization: ${data_audio}/${dset}/${_name} -> ${data_feats}/${dset}/${_name}"

                    # ssl tokenization will additionally need utt2num_samples
                    cp ${data_audio}/${dset}/utt2num_samples ${data_feats}/${dset}
                    scripts/feats/codec_ssl_tokenization.sh \
                        --src_dir ${data_audio}/${dset} \
                        --tgt_dir ${data_feats}/${dset} \
                        --file_name ${_name} \
                        --fs ${fs} \
                        --nj ${nj} \
                        --codec_choice ${codec_choice} \
                        --codec_checkpoint_path ${codec_checkpoint_path} \
                        --codec_config_path ${codec_config_path} \
                        --codec_hf_model_tag ${codec_hf_model_tag} \
                        --codec_batch_size ${codec_batch_size} \
                        --codec_dump_audio false \
                        --ssl_choice ${ssl_choice} \
                        --ssl_checkpoint_path ${ssl_checkpoint_path} \
                        --ssl_kmeans_path ${ssl_kmeans_path} \
                        --ssl_nlayer ${ssl_nlayer} \
                        --ssl_hf_model_tag ${ssl_hf_model_tag} \
                        --ssl_batch_bins ${ssl_batch_bins}

                elif [ ${_modality} == "ssl" ]; then

                    log "ssl tokenization: ${data_audio}/${dset}/${_name} -> ${data_feats}/${dset}/${_name}"

                    # ssl tokenization will additionally need utt2num_samples
                    cp ${data_audio}/${dset}/utt2num_samples ${data_feats}/${dset}
                    scripts/feats/ssl_tokenization.sh \
                        --src_dir ${data_audio}/${dset} \
                        --tgt_dir ${data_feats}/${dset} \
                        --file_name ${_name} \
                        --fs ${fs} \
                        --nj ${nj} \
                        --batch_bins ${ssl_batch_bins} \
                        --ssl_choice ${ssl_choice} \
                        --checkpoint_path ${ssl_checkpoint_path} \
                        --kmeans_path ${ssl_kmeans_path} \
                        --nlayer ${ssl_nlayer} \
                        --hf_model_tag ${ssl_hf_model_tag} \
                        --use_gpu true

                elif [ ${_modality} == "codec" ]; then
                    log "Codec Tokenization: ${data_audio}/${dset}/${_name} -> ${data_feats}/${dset}/${_name}"
                    scripts/feats/codec_tokenization.sh \
                        --src_dir ${data_audio}/${dset} \
                        --tgt_dir ${data_feats}/${dset} \
                        --file_name ${_name} \
                        --codec_fs ${fs} \
                        --dump_audio false \
                        --nj ${nj} \
                        --codec_choice ${codec_choice} \
                        --checkpoint_path ${codec_checkpoint_path} \
                        --config_path ${codec_config_path} \
                        --hf_model_tag ${codec_hf_model_tag}

                elif [ ${_modality} == "g2p" ]; then
                    log "Find G2P vocabulary and copy text"
                    # Use a small portion (up to 100k examples) for efficiency
                    nutt=$(min "100000" "$(wc -l < ${data_audio}/${dset}/${_name})")
                    cat ${data_audio}/${dset}/${_name} | shuf | head -n ${nutt} \
                      > ${data_audio}/${dset}/${_name}.g2p_train && echo ""
                    ${python} -m espnet2.bin.tokenize_text \
                        --token_type "phn" -f 2- \
                        --input "${data_audio}/${dset}/${_name}.g2p_train" \
                        --output "${data_feats}/${dset}/token_lists/g2p_token_list" \
                        --non_linguistic_symbols "${nlsyms_txt}" \
                        --cleaner "${cleaner}" \
                        --g2p "${g2p}" \
                        --write_vocabulary true
                    cp "${data_audio}/${dset}/${_name}" "${data_feats}/${dset}/${_name}"

                elif [ ${_modality} == "text_bpe" ]; then
                    if [ "${subword_choice}" == "huggingface" ]; then
                        if [ -z ${subword_model} ]; then
                            log "Specify hf_tokenizer to use HuggingFace pre-trained tokenizer" && exit 1;
                        fi

                        ${python} pyscripts/utils/build_hf_vocab.py --model_tag ${subword_model} \
                            > ${data_feats}/${dset}/token_lists/text_bpe_token_list
                    else
                        if [ -f ${subword_model}.model ] && [ -f ${subword_model}.vocab ]; then
                            log "Skip training subword model as it already exists: ${subword_model}.model"

                        else
                            if [ -z ${subword_train_text} ]; then
                                subword_train_text=${data_audio}/${dset}/${_name}
                            fi

                            if [ -n "${bpe_nlsyms}" ]; then
                                if test -f "${bpe_nlsyms}"; then
                                    bpe_nlsyms_list=$(awk '{print $1}' ${bpe_nlsyms} | paste -s -d, -)
                                    _opts_spm="--user_defined_symbols=${bpe_nlsyms_list}"
                                else
                                    _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
                                fi
                            else
                                _opts_spm=""
                            fi

                            spm_train \
                                --input=${subword_train_text} \
                                --vocab_size="${nbpe}" \
                                --model_type="${sentencepiece_choice}" \
                                --model_prefix="${data_feats}/${dset}/token_lists/text_bpe" \
                                --character_coverage=${bpe_char_cover} \
                                --input_sentence_size="${bpe_input_sentence_size}" \
                                --shuffle_input_sentence \
                                ${_opts_spm}
                        fi

                        if [ "${dset}" == "${train_set}" ]; then
                            < "${data_feats}/${dset}/token_lists/text_bpe.vocab" awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }' \
                            > ${data_feats}/${dset}/token_lists/text_bpe_token_list
                        fi
                    fi

                    cp ${data_audio}/${dset}/${_name} ${data_feats}/${dset}/${_name}

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

            # The metadata for this dataset/task is saved in a json file
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

if ! ${skip_data_prep}; then
    train_jsons+="${data_feats}/${train_set}/data.json "
    valid_jsons+="${data_feats}/${valid_set}/data.json "
fi

if ! ${skip_train}; then
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

        # Split json files for each data cpu so each data shard is small and easy to handle.
        _logdir="${speechlm_stats_dir}/logdir"
        mkdir -p "${_logdir}"
        ${python} pyscripts/utils/split_data_jsons.py \
            --json_files ${train_jsons} \
            --nj ${nj} \
            --output_dir ${_logdir}/train
        ${python} pyscripts/utils/split_data_jsons.py \
            --json_files ${valid_jsons} \
            --nj ${nj} \
            --output_dir ${_logdir}/valid

        _data_opts=""
        for dset in `ls -d ${_logdir}/train/*/`; do
            _data_opts+="--train_data_path_and_name_and_type ${dset}/split${nj}/JOB/data.JOB.json,_,dataset_json "
        done
        for dset in `ls -d ${_logdir}/valid/*/`; do
            _data_opts+="--valid_data_path_and_name_and_type ${dset}/split${nj}/JOB/data.JOB.json,_,dataset_json "
        done
        _data_opts+="--train_shape_file ${_logdir}/train/example_list.JOB "
        _data_opts+="--valid_shape_file ${_logdir}/valid/example_list.JOB "

        # 2. Generate run.sh
        log "Generate '${speechlm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${speechlm_stats_dir}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${speechlm_stats_dir}/run.sh"; chmod +x "${speechlm_stats_dir}/run.sh"

        # 3. Submit jobs
        log "SpeechLM collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.speechlm_train" \
                --collect_stats true \
                --use_preprocessor true \
                --token_list ${token_list_dir}/token_list \
                --token_bias ${token_list_dir}/token_bias.json \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --subword_choice ${subword_choice} \
                --subword_model "${subword_model}" \
                --multi_task_dataset true \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${_data_opts} ${train_args} \
                || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        _opts+="--skip_sum_stats"
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${speechlm_stats_dir}"

        # (Jinchuan) we only care about the #frames
        for module in enc dec; do
            for dset in train valid; do
                if [ -f ${speechlm_stats_dir}/${dset}/${module}_seq_shape ]; then
                    cat ${speechlm_stats_dir}/${dset}/${module}_seq_shape |\
                    awk -F ',' '{print $1}' \
                    > ${speechlm_stats_dir}/${dset}/${module}_seq_lengths
                fi
            done
        done

        # Shard dataset to each GPU.
        _sharded_dir="${speechlm_stats_dir}/sharded_stats_ngpu${ngpu}"
        mkdir -p "${_sharded_dir}"
        ${python} pyscripts/utils/split_data_jsons.py \
            --json_files ${train_jsons} \
            --nj ${ngpu} \
            --output_dir ${_sharded_dir}/train
        ${python} pyscripts/utils/split_data_jsons.py \
            --json_files ${valid_jsons} \
            --nj ${ngpu} \
            --output_dir ${_sharded_dir}/valid

        for n in `seq $ngpu`; do
            for module in enc dec; do
                for dset in train valid; do
                    if [ -f ${speechlm_stats_dir}/${dset}/${module}_seq_lengths ]; then
                        utils/filter_scp.pl ${_sharded_dir}/${dset}/example_list.${n} \
                            ${speechlm_stats_dir}/${dset}/${module}_seq_lengths \
                            > ${_sharded_dir}/${dset}/${module}_seq_lengths.${n} &
                    fi
                done
            done; wait
        done
    fi

    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: SpeechlLM training: train_jsons=${train_jsons}, valid_set=${valid_jsons}"

        _opts=
        if [ -n "${train_config}" ]; then
            _opts+="--config ${train_config} "
        fi

        _data_opts=""
        _sharded_dir="${speechlm_stats_dir}/sharded_stats_ngpu${ngpu}"

        for dset in `ls -d ${_sharded_dir}/train/*/`; do
            _data_opts+="--train_data_path_and_name_and_type ${dset}/split${ngpu}/JOB/data.JOB.json,_,dataset_json "
        done

        for dset in `ls -d ${_sharded_dir}/valid/*/`; do
            _data_opts+="--valid_data_path_and_name_and_type ${dset}/split${ngpu}/JOB/data.JOB.json,_,dataset_json "
        done

        _data_opts+="--train_shape_file ${_sharded_dir}/train/dec_seq_lengths.JOB "
        _data_opts+="--valid_shape_file ${_sharded_dir}/valid/dec_seq_lengths.JOB "
        if [ -f ${_sharded_dir}/train/enc_seq_lengths.JOB ]; then
            _data_opts+="--train_shape_file ${_sharded_dir}/train/enc_seq_lengths.JOB "
        fi
        if [ -f ${_sharded_dir}/valid/enc_seq_lengths.JOB ]; then
            _data_opts+="--valid_shape_file ${_sharded_dir}/valid/enc_seq_lengths.JOB "
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
                --subword_choice ${subword_choice} \
                --subword_model "${subword_model}" \
                --multi_task_dataset true \
                --sharded_dataset true \
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

        if ! ${skip_data_prep}; then
            for test_set in ${test_sets}; do
                test_jsons+="${data_feats}/${test_set}/data.json "
            done
        fi

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

        for test_json in ${test_jsons}; do
            task=$(grep -o '"task": *[^,}]*' ${test_json} | sed -e 's/"task": *//' -e 's/"//g')
            dset=$(basename $(dirname "${test_json}"))
            _dir="${speechlm_exp}/${inference_tag}/${task}_${dset}"
            _logdir="${_dir}/log"
            mkdir -p ${_logdir}

            ${python} pyscripts/utils/split_data_jsons.py \
                --json_files ${test_json} \
                --nj ${inference_nj} \
                --output_dir ${_logdir}

            _data_opts="--data_path_and_name_and_type ${_logdir}/${dset}/split${inference_nj}/JOB/data.JOB.json,_,dataset_json"

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/speechlm_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${inference_nj}" "${_logdir}"/speechlm_inference.JOB.log \
                ${python} -m espnet2.bin.speechlm_inference \
                    --ngpu "${_ngpu}" \
                    --nbest ${nbest} \
                    --model_file "${speechlm_exp}"/"${inference_model}" \
                    --train_config "${speechlm_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${_data_opts} ${inference_args} \
                    || { cat $(grep -l -i error "${_logdir}"/speechlm_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for entry in `ls ${_logdir}/output.1`; do
                if [ -f ${_logdir}/output.1/${entry}/${entry} ]; then
                    for n in `seq ${inference_nj}`; do
                        cat ${_logdir}/output.${n}/${entry}/${entry}
                    done | sort > ${_dir}/${entry}
                fi

                for n in `seq ${inference_nj}`; do
                    cat ${_logdir}/output.${n}/${entry}/token_${entry}.scp
                done | sort > ${_dir}/token_${entry}.scp
            done
        done
    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Evaluating the model ..."

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        if ! ${skip_data_prep}; then
            for test_set in ${test_sets}; do
                test_jsons+="${data_feats}/${test_set}/data.json "
            done
        fi

        for test_json in ${test_jsons}; do
            # (1) Find task, dataset name and folder name
            task=$(grep -o '"task": *[^,}]*' ${test_json} | sed -e 's/"task": *//' -e 's/"//g')
            _src_dir="$(dirname "${test_json}")"
            _dset="$(basename ${_src_dir})"
            _dir="${speechlm_exp}/${inference_tag}/${task}_${_dset}";
            mkdir -p ${_dir}/eval_cache

            target_triplets=$(python -c "from espnet2.speechlm.definitions import SPEECHLM_TASKS; print(SPEECHLM_TASKS['${task}'].target_string)")
            target_files=$(echo ${target_triplets} | tr ' ' '\n' | awk -F ',' '{print $1}')

            condition_triplets=$(python -c "from espnet2.speechlm.definitions import SPEECHLM_TASKS; print(SPEECHLM_TASKS['${task}'].condition_string)")
            condition_files=$(echo ${condition_triplets} | tr ' ' '\n' | awk -F ',' '{print $1}')

            # (2) process files before scoring.
            # (2.1) intersection of all generated example files

            awk '{print $1}' ${_dir}/$(echo ${target_files} | cut -d ' ' -f 1) > ${_dir}/eval_cache/gen_list
            for file in $(echo ${target_files} | cut -d ' ' -f 2-); do
                utils/filter_scp.pl ${_dir}/eval_cache/gen_list ${_dir}/${file} \
                    > ${_dir}/eval_cache/gen_list.tmp
                mv ${_dir}/eval_cache/gen_list.tmp ${_dir}/eval_cache/gen_list
            done

            # (2.2) extend the prefix and suffix of all reference files. I.e.,
            #       all target files in decoding dir;
            #       all condition files in original data dir
            all_ref_files=
            for file in ${condition_files}; do
                if [ -f ${_dir}/${file} ]; then
                    all_ref_files+="${_dir}/${file} "
                fi
            done
            for file in ${target_files}; do
                all_ref_files+="${_src_dir}/index_files/${file} "
            done

            if [ "${task}" == "tts" ]; then
                all_ref_files+="${_src_dir}/index_files/text "
            fi

            for file in ${all_ref_files}; do
                name=$(basename ${file})
                awk -v N=${nbest} -v Task=${task} '{{name=$1}for(i=0; i<N; i++){$1=Task "_" name "_sample" i; print $0}}' \
                    ${file} > ${_dir}/eval_cache/${name}

                utils/filter_scp.pl ${_dir}/eval_cache/gen_list ${_dir}/eval_cache/${name} \
                    > ${_dir}/eval_cache/${name}.tmp
                mv ${_dir}/eval_cache/${name}.tmp ${_dir}/eval_cache/${name}
            done

            # (2.3) generated valid keys
            for file in ${target_files}; do
                awk -v prefix="${task}" '{print prefix "_" $1}' ${_src_dir}/index_files/${file}
            done | sort | uniq > ${_dir}/eval_cache/key_file

            # (3) Task-specific evaluation
            ./scripts/utils/speechlm_eval/eval_${task}.sh \
                --gen_dir ${_dir} \
                --ref_dir ${_dir}/eval_cache \
                --key_file ${_dir}/eval_cache/key_file \
                --nj ${nj} \
                --inference_nj ${inference_nj} \
                --gpu_inference ${gpu_inference} \
                --nbest ${nbest} ${scoring_args}
        done
    fi
else
    log "Skip the evaluation stages"
fi

packed_model="${speechlm_exp}/${speechlm_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Pack model: ${packed_model}"

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack speechlm \
        --train_config "${speechlm_exp}"/config.yaml \
        --inference_config "${inference_config}" \
        --model_file "${speechlm_exp}"/"${inference_model}" \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1
    log "Stage 15: Upload model to HuggingFace: ${hf_repo}"

    if [ ! -f "${packed_model}" ]; then
        log "ERROR: ${packed_model} does not exist. Please run stage 11 first."
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
    # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=Speech-Language-Model
    # shellcheck disable=SC2034
    espnet_task=SpeechLM
    # shellcheck disable=SC2034
    lang=multilingual
    task_exp=${speechlm_exp}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
    echo done
fi

# TODO(Jinchuan) Upload the prepared data and trained models
if ! "${skip_upload_hf_data}"; then
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "upload the prepared ${task} dataset prepared in ${data_feats}"

        if [ "$(huggingface-cli whoami)" == "Not logged in" ]; then
            echo "You should login huggingface-cli before uploading the dataset" && exit 1;
        fi

        huggingface-cli repo create -y ${data_combo_name} --type dataset
        huggingface-cli upload --repo-type dataset ${data_combo_name} \
            ${data_feats} ${data_feats} \
            --exclude "*.log"
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
