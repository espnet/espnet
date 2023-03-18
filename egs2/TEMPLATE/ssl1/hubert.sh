#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# Copyright 2021 Tianzi Wang
#           2022 Xuankai Chang
# Apache 2.0
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

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
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=word      # Tokenization type (char or bpe).

# Pretrain model related
hubert_args=         # Arguments for asr model training, e.g., "--max_epoch 10".
                     # Note that it will overwrite args in asr config.
num_splits_ssl=1 # Number of splitting for lm corpus.

# Pretrain related
train_start_iter= # Pretrain starts from the specified iteration (0 mean MFCC iteraion)
train_stop_iter=  # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion)
train_configs=    # Configration files of training stage
feats_normalize=  # Normalizaton layer type.
n_clusters=             # Number of k-means clusters (multiple values, e.g. "100 500 500")
features_km=            # Feature for k-means clustering (multiple values, e.g. "mfcc hubert hubert")
layers_km=              # Layers of feature for k-means clustering of training stage (multiple values, e.g. "0 6 9")
portion_km=1            # Portion of training set used for k-means
gpu_dump_feature=false  # Whether to use gpu in kmeans process for feature dumping.
alignment_phoneme_dir=  # Phoneme alignment directory with tsv file (utt_id, phoneme_alignment)

# Upload model related
hf_repo=
inference_ssl_model=valid.loss.best.pth # SSL model path from previous iteration and uploading

# [Task dependent] Set the datadir name created by local/data.sh
train_set=     # Name of training set
valid_set=     # Name of valid set
lang=noinfo    # The language type of corpus.
speech_fold_length=800 # fold_length for speech data during SSL training.
text_fold_length=400   # fold_length for text data during SSL training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}")
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload_hf # Skip packing and uploading stages (default="${skip_upload_hf}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").
    --hf_repo        # Hugging face repository name (default="${hf_repo}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").

    # HuBERT model related
    --num_splits_ssl   # Number of splitting for ssl training  (default="${num_splits_ssl}").

    # HuBERT related
    --train_start_iter  # Pretrain starts from the specified iteration (0 mean MFCC iteraion, default="${train_start_iter}").
    --train_stop_iter   # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion, default="${train_stop_iter}").
    --train_configs    # configration files of training stage
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --n_clusters       # number of k-means clusters of training stages (e.g. "100 500 500")
    --features_km      # feature for k-means clustering of training stages (e.g. "mfcc hubert hubert")
    --layers_km        # layers of feature for k-means clustering of training stages (e.g. "0 6 9")
    --hubert_args      # Arguments for hubert model training (default="${hubert_args}").
                       # e.g., --hubert_args "--max_epoch 10"
                       # Note that it will overwrite args in pt config.
    --gpu_dump_feature # Whether to use gpu for kmeans feature dumping (default="${gpu_dump_feature}").

    # Alignment
    --alignment_phoneme_dir # Phoneme alignment directory with tsv file (utt_id, phoneme_alignment)

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training train set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --lang          # The language type of corpus (default=${lang}).
    --speech_fold_length  # fold_length for speech data during HuBERT training (default="${speech_fold_length}").
    --text_fold_length    # fold_length for text data during HuBERT training (default="${text_fold_length}").
    --inference_ssl_model # SSL model path from previous iteration and uploading (default="${inference_ssl_model}").
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

# Check train_configs, n_clusters, features, layers and their lengths
train_config_list=(${train_configs// / })
if [ ${#train_config_list[@]} -le ${train_stop_iter} ]; then
    log "Error: # train_configs ${#train_config_list[@]} is less than train_stop_iter ${train_stop_iter}"
    exit 1;
fi
n_clusters_list=(${n_clusters// / })
if [ ${#n_clusters_list[@]} -le ${train_stop_iter} ]; then
    log "Error: # n_clusters_list ${#n_clusters_list[@]} is less than train_stop_iter ${train_stop_iter}"
    exit 1;
fi
feature_list=(${features_km// / })
if [ ${#feature_list[@]} -le ${train_stop_iter} ]; then
    log "Error: # feature_list ${#feature_list[@]} is less than train_stop_iter ${train_stop_iter}"
    exit 1;
fi
layer_list=(${layers_km// / })
if [ ${#layer_list[@]} -le ${train_stop_iter} ]; then
    log "Error: # layer_list ${#layer_list[@]} is less than train_stop_iter ${train_stop_iter}"
    exit 1;
fi
if ! [ ${train_start_iter} -le ${train_stop_iter} ]; then
    log "Error: train_start_iter is required to be smaller or equal than train_stop_iter"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
            log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    scripts/utils/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                    _dirs+="data/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="data/${train_set} "
                fi
            done
            utils/combine_data.sh "data/${train_set}_sp" ${_dirs}
        else
            log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}"; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                                                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                                                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
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
            else
                log "Error: not supported: --feats_type ${feats_type}"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
            awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the stages for data preparation"
fi

if ! "${skip_train}"; then

    for ((iter=${train_start_iter}; iter<=${train_stop_iter};iter++)); do

        # Get the feature type, feature layer, n_clusters and config for the current iteration
        feats_km="${feature_list[${iter}]}"
        layer="${layer_list[${iter}]}"
        n_clusters="${n_clusters_list[${iter}]}"
        ssl_config="${train_config_list[${iter}]}"

        if [ -n "${ssl_config}" ]; then
            ssl_tag="$(basename "${ssl_config}" .yaml)_${feats_type}"
        else
            ssl_tag="train_${feats_type}"
        fi

        ssl_stats_dir="${expdir}/hubert_iter${iter}_stats_${feats_type}"
        ssl_exp="${expdir}/hubert_iter${iter}_${ssl_tag}"
        token_listdir="data/${lang}_token_list_kmeans_iter${iter}_${feats_km}_${n_clusters}clusters/${token_type}"
        km_tag="kmeans_iter${iter}_${feats_km}_${train_set}_portion${portion_km}"

        if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
            log "Stage 5 [Iter ${iter} / ${train_stop_iter}]: Running ${n_clusters} cluster K-means on ${feats_km} feature."

            _opts=
            if [ ${iter} -ge 1 ]; then
                if ! "${gpu_dump_feature}"; then
                    log "Warning: It is recommented to use GPU in HuBERT feature extraction for K-means clustering."
                fi
                _opts+="--use_gpu ${gpu_dump_feature} "

                pretrained_ssl_tag="$(basename "${train_config_list[${iter}-1]}" .yaml)_${feats_type}"
                pretrained_ssl_exp="${expdir}/hubert_iter$(( iter-1 ))_${pretrained_ssl_tag}"

                _opts+="--hubert_url espnet "
                _opts+="--hubert_dir_path ${pretrained_ssl_exp}/${inference_ssl_model} "
            fi

            ./local/perform_kmeans.sh \
                --stage 1 \
                --stop_stage 5 \
                --train_set "${train_set}" \
                --dev_set "${valid_set}" \
                --nclusters "${n_clusters}" \
                --feature_type "${feats_km}" \
                --layer "${layer}" \
                --datadir "${data_feats}" \
                --feat_dir "${dumpdir}/hubert_feats" \
                --km_dir "${expdir}/${km_tag}" \
                --portion "${portion_km}" \
                --dictdir "${token_listdir}" \
                --nj ${nj} \
                ${alignment_phoneme_dir:+--alignment_phoneme_dir ${alignment_phoneme_dir}} \
                ${_opts} || exit 1;
        fi

        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            _ssl_train_dir="${data_feats}/${train_set}"
            _ssl_valid_dir="${data_feats}/${valid_set}"

            log "Stage 6 [Iter ${iter} / ${train_stop_iter}]: HuBERT collect stats: input_feats=${feats_km} train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"

            _opts=
            if [ -n "${ssl_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.hubert_train --print_config --optim adam
                _opts+="--config ${ssl_config} "
            fi

            _feats_type="$(<${_ssl_train_dir}/feats_type)"
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
                _scp=feats.scp
                _type=kaldi_ark
                _input_size="$(<${_ssl_train_dir}/feats_dim)"
                _opts+="--input_size=${_input_size} "
            fi

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
            log "Generate '${ssl_stats_dir}/run.sh'. You can resume the process from stage 5.iter${iter} using this script"
            mkdir -p "${ssl_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${ssl_stats_dir}/run.sh";chmod +x "${ssl_stats_dir}/run.sh"

            # 3. Submit jobs
            log "HuBERT collect-stats started... log: '${_logdir}/stats.*.log'"

            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.

            # shellcheck disableSC2046,SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.hubert_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --normalize none \
                    --token_type "${token_type}" \
                    --token_list "${token_listdir}/tokens.txt" \
                    --num_classes "${n_clusters}" \
                    --train_data_path_and_name_and_type "${_ssl_train_dir}/${_scp},speech,${_type}" \
                    --train_data_path_and_name_and_type "${_ssl_train_dir}/text.km.${km_tag},text,text" \
                    --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
                    --valid_data_path_and_name_and_type "${_ssl_valid_dir}/text.km.${km_tag},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/valid.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${hubert_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            if [ "${feats_normalize}" != global_mvn ]; then
                # Skip summerizaing stats if not using global MVN
                _opts+="--skip_sum_stats"
            fi
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${ssl_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${ssl_stats_dir}/train/text_shape" \
                awk -v N="$(<${token_listdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
                >"${ssl_stats_dir}/train/text_shape.${token_type}"

            <"${ssl_stats_dir}/valid/text_shape" \
                awk -v N="$(<${token_listdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
                >"${ssl_stats_dir}/valid/text_shape.${token_type}"
        fi

        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            _ssl_train_dir="${data_feats}/${train_set}"
            _ssl_valid_dir="${data_feats}/${valid_set}"

            log "Stage 7 [Iter ${iter} / ${train_stop_iter}]: HuBERT Training: input_feats=${feats_km}, train_set=${_ssl_train_dir}, valid_set=${_ssl_valid_dir}"

            _opts=
            if [ -n "${ssl_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.hubert_train --print_config --optim adam
                _opts+="--config ${ssl_config} "
            fi

            _feats_type="$(<${_ssl_train_dir}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                # "sound" supports "wav", "flac", etc.
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
                _fold_length="$((speech_fold_length * 100))"
                _opts+="--frontend_conf fs=${fs} "
            else
                _scp=feats.scp
                _type=kaldi_ark
                _fold_length="${speech_fold_length}"
                _input_size="$(<${_ssl_train_dir}/feats_dim)"
                _opts+="--input_size=${_input_size} "
            fi
            if [ "${feats_normalize}" = global_mvn ]; then
                # Default normalization is utterance_mvn and changes to global_mvn
                _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
            fi

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
                              "${_ssl_train_dir}/text" \
                              "${ssl_stats_dir}/train/speech_shape" \
                              "${ssl_stats_dir}/train/text_shape.${token_type}" \
                              --num_splits "${num_splits_ssl}" \
                              --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.km.${km_tag},text,text "
                _opts+="--train_shape_file ${_split_dir}/speech_shape "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/text.km.${km_tag},text,text "
                _opts+="--train_shape_file ${ssl_stats_dir}/train/speech_shape "
                _opts+="--train_shape_file ${ssl_stats_dir}/train/text_shape.${token_type} "
            fi

            log "Generate '${ssl_exp}/run.sh'. You can resume the process from stage 7 using this script"
            mkdir -p "${ssl_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${ssl_exp}/run.sh"; chmod +x "${ssl_exp}/run.sh"

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
            log "HuBERT Training started... log: '${ssl_exp}/train.log'"
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
                ${python} -m espnet2.bin.hubert_train \
                    --use_preprocessor true \
                    --normalize null \
                    --token_type "${token_type}" \
                    --token_list "${token_listdir}/tokens.txt" \
                    --num_classes "${n_clusters}" \
                    --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
                    --valid_data_path_and_name_and_type "${_ssl_valid_dir}/text.km.${km_tag},text,text" \
                    --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
                    --valid_shape_file "${ssl_stats_dir}/valid/text_shape.${token_type}" \
                    --resume true \
                    --fold_length "${_fold_length}" \
                    --fold_length "${text_fold_length}" \
                    --output_dir "${ssl_exp}" \
                    ${_opts} ${hubert_args}

            log "Iter ${iter} HuBERT Training completed, model saved in: ${ssl_exp}"
        fi
    done
else
    log "Skip the training stages"
fi

if [ -n "${ssl_config}" ]; then
    ssl_tag="$(basename "${ssl_config}" .yaml)_${feats_type}"
else
    ssl_tag="train_${feats_type}"
fi
ssl_exp="${expdir}/hubert_iter${train_stop_iter}_${ssl_tag}"
km_tag="kmeans_iter${train_stop_iter}_${feature_list[${train_stop_iter}]}_${train_set}_portion${portion_km}"
packed_model="${ssl_exp}/${ssl_exp##*/}_${inference_ssl_model%.*}.zip"
# Skip pack preparation if using a downloaded model
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Pack model: ${packed_model}"

    _opts=
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${asr_stats_dir}/train/feats_stats.npz "
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack ssl \
        --train_config "${ssl_exp}"/config.yaml \
        --model_file "${ssl_exp}"/"${inference_ssl_model}" \
        ${_opts} \
        --option "${ssl_exp}"/images \
        --option "${expdir}/${km_tag}/km_${n_clusters_list[${train_stop_iter}]}.mdl" \
        --outpath "${packed_model}"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
        exit 1
        log "Stage 9: Upload model to HuggingFace: ${hf_repo}"

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
        hf_task=self-supervised-learning
        # shellcheck disable=SC2034
        espnet_task=SSL
        # shellcheck disable=SC2034
        task_exp=${ssl_exp}
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
    log "Skip the uploading to HuggingFace stage"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
