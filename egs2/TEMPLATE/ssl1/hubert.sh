#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# Copyright 2021 Tianzi Wang
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
pretrain_start_iter= # Pretrain starts from the specified iteration (0 mean MFCC iteraion)
pretrain_stop_iter=  # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion)
skip_data_prep=false # Skip data preparation stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1      # The number of gpus in pretrain stage ("0" uses cpu, otherwise use gpu).
num_nodes=1 # The number of nodes in pretrain stage.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
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
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
pad="<pad>"         # pad symbol
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Pretrain model related
pt_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
num_splits_asr=1           # Number of splitting for lm corpus.

# Pretrain related
n_clusters=                # Number of k-means clusters of pretraining stage
features_km=               # Feature for k-means clustering of pretraining stage
portion_km=                # Portion of training set used for k-means
pretrain_configs=          # Configration files of pretraining stage

download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=     # Name of pretrain training set
valid_set=     # Name of pretraining valid set
bpe_train_text= # Text file path of bpe training set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
asr_speech_fold_length=800 # fold_length for speech data during ASR training.
asr_text_fold_length=150   # fold_length for text data during ASR training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage                # Processes starts from the specified stage (default="${stage}").
    --stop_stage            # Processes is stopped at the specified stage (default="${stop_stage}").
    --pretrain_start_iter  # Pretrain starts from the specified iteration (0 mean MFCC iteraion, default="${pretrain_start_iter}").
    --pretrain_stop_iter   # Pretrain is stopped from the specified iteration (0 mean MFCC iteraion, default="${pretrain_stop_iter}").
    --skip_data_prep        # Skip data preparation stages (default="${skip_data_prep}").
    --skip_eval             # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload           # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu                 # The number of gpus in pretrain stage ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes            # The number of nodes in pretrain stage (default="${num_nodes}").
    --nj                    # The number of parallel jobs (default="${nj}").
    --dumpdir               # Directory to dump features (default="${dumpdir}").
    --expdir                # Directory to save experiments (default="${expdir}").
    --python                # Specify python to execute espnet commands (default="${python}").
    # Data preparation related
    --local_data_opts       # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format            # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs                      # Sampling rate (default="${fs}").
    --min_wav_duration        # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration        # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Pretrain related
    --pretrain_configs # configration files of pretraining stage
    --n_clusters       # number of k-means clusters of pretraining stage
    --features_km      # feature for k-means clustering of pretraining stage
    --pt_args         # Arguments for hubert model pretraining (default="${pt_args}").
                       # e.g., --pt_args "--max_epoch 10"
                       # Note that it will overwrite args in pt config.

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of pretraining train set
    --valid_set     # Name of pretraining valid set
    --bpe_train_text # Text file path of bpe training set.
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --asr_speech_fold_length # fold_length for speech data during ASR training (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training (default="${asr_text_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh
echo $@
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

# Check pretrain_config, n_clusters and feature list
pretrain_config_list=(${pretrain_configs// / })
n_clusters_list=(${n_clusters// / })
feature_list=(${features_km// / })
if ! [ ${pretrain_start_iter} -le ${pretrain_stop_iter} ]; then
    log "Error: pretrain_start_iter is required to be smaller or equal than pretrain_stop_iter"
fi

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation"
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
	    _suf="/org"
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


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 5 ]; then

    for ((iter=${pretrain_start_iter}; iter<=${pretrain_stop_iter};iter++)); do
        asr_config="${pretrain_config_list[${iter}]}"
        if [ "${lang}" != noinfo ]; then
            asr_stats_dir="${expdir}/pretrain_iter${iter}_stats_${feats_type}_${lang}"
        else
            asr_stats_dir="${expdir}/pretrain_iter${iter}_stats_${feats_type}"
        fi

        if [ -n "${asr_config}" ]; then
            asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}"
        else
            asr_tag="train_${feats_type}"
        fi

        asr_exp="${expdir}/pretrain_${asr_tag}_iter${iter}"

        train_set_plabel=$(eval "echo ${train_set}_\${feature_list[${iter}]}_km\${n_clusters_list[${iter}]}")
        valid_set_plabel=$(eval "echo ${valid_set}_\${feature_list[${iter}]}_km\${n_clusters_list[${iter}]}")

        feats_km="${feature_list[${iter}]}"
        n_clusters="${n_clusters_list[${iter}]}"
        dictdir="./data/${feats_km}_km${n_clusters}_token_list_iter${iter}/${token_type}"

        if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
            log "Stage 5.iter${iter}: Running ${n_clusters} cluster K-means on ${feats_km} feature."

            if [ ${iter} -eq 0 ] || [ ${feats_km} == "mfcc" ]; then
                ./scripts/km.sh \
                    --train_set "${train_set}" \
                    --dev_set "${valid_set}" \
                    --nclusters "${n_clusters}" \
                    --feature-type "${feats_km}" \
                    --datadir "${data_feats}" \
                    --kmrootdir "${expdir}" \
                    --portion "${portion_km}" \
                    --dictdir "${dictdir}"
            else
                ./scripts/km.sh \
                    --train_set "${train_set}" \
                    --dev_set "${valid_set}" \
                    --nclusters "${n_clusters}" \
                    --feature-type "${feats_km}" \
                    --datadir "${data_feats}" \
                    --kmrootdir "${expdir}" \
                    --portion "${portion_km}" \
                    --dictdir "${dictdir}" \
                    --hubert_url espnet \
                    --hubert_dir_path "${expdir}/pretrained_model_iter$((iter-1))"/valid.acc.best.pth
            fi
        fi

        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            _asr_train_dir="${data_feats}/${train_set_plabel}"
            _asr_valid_dir="${data_feats}/${valid_set_plabel}"

            log "Stage 6.iter${iter}: ${feats_km} pretrain model collect stats: \
                       train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

            _opts=
            if [ -n "${asr_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
                _opts+="--config ${asr_config} "
            fi

            _feats_type="$(<${_asr_train_dir}/feats_type)"
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
                _input_size="$(<${_asr_train_dir}/feats_dim)"
                _opts+="--input_size=${_input_size} "
            fi

            # 1. Split the key file
            _logdir="${asr_stats_dir}/logdir"
            mkdir -p "${_logdir}"

            # Get the minimum number among ${nj} and the number lines of input files
            _nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_valid_dir}/${_scp} wc -l)")

            key_file="${_asr_train_dir}/${_scp}"
            split_scps=""
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/train.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            key_file="${_asr_valid_dir}/${_scp}"
            split_scps=""
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/valid.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Generate run.sh
            log "Generate '${asr_stats_dir}/run.sh'. You can resume the process from stage 5.iter${iter} using this script"
            mkdir -p "${asr_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${asr_stats_dir}/run.sh"; chmod +x "${asr_stats_dir}/run.sh"

            # 3. Submit jobs
            log "Hubert pretraining collect-stats started... log: '${_logdir}/stats.*.log'"

            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.

            # shellcheck disableSC2046,SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                         ${python} -m espnet2.bin.hubert_train \
                         --collect_stats true \
                         --use_preprocessor true \
                         --normalize none \
                         --bpemodel none \
                         --token_type "${token_type}" \
                         --token_list "${dictdir}/tokens.txt" \
                         --non_linguistic_symbols none \
                         --cleaner "${cleaner}" \
                         --g2p "${g2p}" \
                         --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
                         --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
                         --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                         --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                         --train_shape_file "${_logdir}/train.JOB.scp" \
                         --valid_shape_file "${_logdir}/valid.JOB.scp" \
                         --output_dir "${_logdir}/stats.JOB" \
                         --hubert_dict "${dictdir}/dict.txt" \
                         ${_opts} ${pt_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${asr_stats_dir}/train/text_shape" \
             awk -v N="$(<${dictdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
             >"${asr_stats_dir}/train/text_shape.${token_type}"

            <"${asr_stats_dir}/valid/text_shape" \
             awk -v N="$(<${dictdir}/tokens.txt wc -l)" '{ print $0 "," N }' \
             >"${asr_stats_dir}/valid/text_shape.${token_type}"
        fi

        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            _asr_train_dir="${data_feats}/${train_set_plabel}"
            _asr_valid_dir="${data_feats}/${valid_set_plabel}"

            log "Stage 7.iter${iter}: Hubert Pretraining: train_set=${_asr_train_dir}, valid_set=${_asr_valid_dir}"

            _opts=
            if [ -n "${asr_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.hubert_train --print_config --optim adam
                _opts+="--config ${asr_config} "
            fi

            _feats_type="$(<${_asr_train_dir}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                # "sound" supports "wav", "flac", etc.
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
                _fold_length="$((asr_speech_fold_length * 100))"
                _opts+="--frontend_conf fs=${fs} "
            else
                _scp=feats.scp
                _type=kaldi_ark
                _fold_length="${asr_speech_fold_length}"
                _input_size="$(<${_asr_train_dir}/feats_dim)"
                _opts+="--input_size=${_input_size} "
            fi

            if [ "${num_splits_asr}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                              --scps \
                              "${_asr_train_dir}/${_scp}" \
                              "${_asr_train_dir}/text" \
                              "${asr_stats_dir}/train/speech_shape" \
                              "${asr_stats_dir}/train/text_shape.${token_type}" \
                              --num_splits "${num_splits_asr}" \
                              --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
                _opts+="--train_shape_file ${_split_dir}/speech_shape "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
                _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
                _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
            fi

            log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${asr_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
            log "Hubert pretraining started... log: '${asr_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${asr_exp})"
            else
                jobname="${asr_exp}/train.log"
            fi

            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                      --cmd "${cuda_cmd} --name ${jobname}" \
                      --log "${asr_exp}"/train.log \
                      --ngpu "${ngpu}" \
                      --num_nodes "${num_nodes}" \
                      --init_file_prefix "${asr_exp}"/.dist_init_ \
                      --multiprocessing_distributed false -- \
                      ${python} -m espnet2.bin.hubert_train \
                      --use_preprocessor true \
                      --normalize null \
                      --bpemodel none \
                      --token_type "${token_type}" \
                      --token_list "${dictdir}/tokens.txt" \
                      --non_linguistic_symbols none \
                      --cleaner "${cleaner}" \
                      --g2p "${g2p}" \
                      --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                      --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                      --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                      --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
                      --resume true \
                      --fold_length "${_fold_length}" \
                      --fold_length "${asr_text_fold_length}" \
                      --output_dir "${asr_exp}" \
                      --hubert_dict "${dictdir}/dict.txt" \
                      ${_opts} ${pt_args}

            if [ "${iter}" -ge 0 ]; then
                log "Create a symbolic link of the pretrained model"
                if  [ -L "${expdir}/pretrained_model_iter${iter}" ]; then
                    log "Symbolic link ${expdir}/pretrained_model_iter${iter} already exists, remove it."
                    rm "${expdir}/pretrained_model_iter${iter}"
                fi

                if ! [ -z "${asr_exp}" ]; then
                    ln -s "../${asr_exp}" "${expdir}/pretrained_model_iter${iter}"
                fi
            fi

            log "Model saved in: ${asr_exp}"
        else
            log "Skip the pretraining stages"
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
