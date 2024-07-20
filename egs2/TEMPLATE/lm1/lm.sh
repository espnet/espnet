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
stage=1                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_stages=            # Processes is stopped at the specified stage.
skip_data_prep=false    # Skip data preparation stages
skip_train=false        # Skip training stages
skip_eval=false         # Skip decoding and evaluation stages
skip_packing=true       # Skip the packing stage.
skip_upload_hf=true     # Skip uploading to huggingface stage.
ngpu=1                  # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1             # The number of nodes
nj=32                   # The number of parallel jobs.
dumpdir=dump            # Directory to dump features.
inference_nj=4          # The number of parallel jobs in decoding.
gpu_inference=false     # Whether to perform gpu decoding.
expdir=exp              # Directory to save experiments.
python=python3          # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw, or extracted).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Kmeans related
km_dir=                     # Path to pretrained kmeans model
learn_kmeans=true           # boolean flag to note whether to learn kmeans
kmeans_opts=                # The options given to scripts/feats/perform_kmeans.sh, needed when kmeans is trained
kmeans_feature="hubert_base/6" # format: ssl_model_type/layer_idx (e.g. mfcc, hubert_large/21, wavlm_large/21), needed when kmeans is trained
portion=0.1
nclusters=1000              # The number of clusters for discrete tokens, needed when kmeans is trained
storage_save_mode=true      # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
token_case="ts"
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
bpe_char_cover=1.0  # character coverage when modeling BPE for source language
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma or a file containing 1 symbol per line, for BPE

# Language model related
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
num_splits_lm=1   # Number of splitting for lm corpus.

# Decoding related
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.acc.ave.pth       # Language model path for decoding.
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_test_text_asr=    # Text file path of asr evaluation set.
lm_test_text_tts=    # Text file path of tts evaluation set.
lm_test_text_textlm="dummy"    # Text file path of textlm evaluation set.
lm_test_text_speechlm="dummy"    # Text file path of unitlm evaluation set.
lm_inference_asr_config=    # Config for decoding asr.
lm_inference_tts_config=    # Config for decoding tts.
lang=noinfo      # The language type of corpus.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lm_fold_length=150      # fold_length for LM training.
# Language Model specific parameters
use_speech=true
use_text=true

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"
Options:
    # General configuration
    --stage              # Processes starts from the specified stage (default="${stage}").
    --stop_stage         # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_stages        # Spicify the stage to be skipped (default="${skip_stages}").
    --skip_data_prep     # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train         # Skip training stages (default="${skip_train}").
    --skip_eval          # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_packing       # Skip the packing stage (default="${skip_packing}").
    --skip_upload_hf     # Skip uploading to huggingface stage (default="${skip_upload_hf}").
    --ngpu               # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes          # The number of nodes (default="${num_nodes}").
    --nj                 # The number of parallel jobs (default="${nj}").
    --inference_nj       # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference      # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir            # Directory to dump features (default="${dumpdir}").
    --expdir             # Directory to save experiments (default="${expdir}").
    --python             # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (raw, or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw or raw_copy, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --token_case              # Token case type: ts: true sequence rm: remove repitions.
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma or a file containing 1 symbol per line . (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Kmeans related
    --km_dir            # Path to pretrained kmeans model
    --learn_kmeans      # boolean flag to note whether to learn kmeans (default=false).
    --kmeans_opts       # The options given to kmeans step (default="${kmeans_opts}").
    --kmeans_feature    # The string indicates the kmeans features (default="${kmeans_feature}").
    --portion           # The portion of data used to train kmeans (default="${portion}").
    --nclusters         # The number of clusters for discrete tokens (default="${nclusters}").
    --storage_save_mode # # Save storage on SSL feature extraction. If true, feature extraction and kmeans clustering on the fly (default="${storage_save_mode}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

   # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_test_text_asr    # Text file path of asr evaluation set.
    --lm_test_text_tts    # Text file path of tts evaluation set.
    --lm_test_text_textlm    # Text file path of textlm evaluation set.
    --lm_test_text_speechlm    # Text file path of unitlm evaluation set.
    --lm_inference_asr_config    # Config for decoding asr.
    --lm_inference_tts_config    # Config for decoding tts.
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --lang          # The language type of corpus (default=${lang}).
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").

    # Language Model specific parameters
    --use_speech    # Whether to use speech for langauge model
    --use_text      # Whether to use text for langauge model

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
if ! "${skip_train}"; then
    [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi
if [ -n "${train_set}" ] && [ "${train_set}" = "${valid_set}" ]; then
    log "Error: train_set and valid_set must be different. --train_set ${train_set} --valid_set ${valid_set}"
    exit 1
fi
_test_sets=
for dset in ${test_sets}; do
    if [ "${dset}" = "${train_set}" ]; then
        log "Error: train_set and test_sets must be different. --train_set ${train_set} --test_sets ${test_sets}"
        exit 1
    fi
    if [ "${dset}" = "${valid_set}" ]; then
        log "Info: The valid_set '${valid_set}' is included in the test_sets. '--eval_valid_set true' is set and '${valid_set}' is removed from the test_sets"
        eval_valid_set=true
    elif [[ " ${_test_sets} " =~ [[:space:]]${dset}[[:space:]] ]]; then
        log "Info: ${dset} is duplicated in the test_sets. One is removed"
    else
        _test_sets+="${dset} "
    fi
done
test_sets=${_test_sets}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_audio=${dumpdir}/audio_raw
    data_extract=${dumpdir}/extracted
    data_feats=${dumpdir}/"${feats_type}"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

lm_train_text="${data_feats}/${train_set}/lm_text"
lm_dev_text="${data_feats}/${valid_set}/lm_text"

[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/org/${train_set}/text"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi

bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi

if [ ${kmeans_feature} = "mfcc" ]; then  # MFCC has no layer
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=
    kmeans_feature_conf="{type=mfcc}"
else
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
    s3prl_conf="{upstream=${kmeans_feature_type}}"
    kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"
fi
if [ -z "${km_dir}" ]; then
    km_dir="${expdir}"/kmeans/$(echo "${kmeans_feature}" | tr "/" "_")_${nclusters}clusters
fi

if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${token_type}"
    else
        lm_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi

if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi

if "${skip_data_prep}"; then
    skip_stages+="1 2 3 4 5 "
fi
if "${skip_train}"; then
    skip_stages+="2 4 5 6 7 "
fi
if "${skip_eval}"; then
    skip_stages+="8 9 10 "
fi

if "${skip_packing}"; then
    skip_stages+="11 "
fi
if "${skip_upload_hf}"; then
    skip_stages+="12 "
fi

skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if "${skip_train}"; then
        _dsets="${test_sets}"
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi
    if "${use_speech}"; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 2: Format wav.scp: data/ -> ${data_audio}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in ${_dsets}; do
                echo $dset
                for _dir in "data/${dset}/speech/"*; do
                    echo ${_dir}
                    if [ -d "${_dir}" ]; then
                        echo "${_dir}"   # your processing here

                        utils/copy_data_dir.sh --validate_opts --non-print "${_dir}" "${data_audio}/$(basename ${_dir})/${dset}/"

                        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                            --audio-format "${audio_format}" --fs "${fs}" \
                            "${_dir}/wav.scp" "${data_audio}/$(basename ${_dir})/${dset}"

                        echo "${feats_type}" > "${data_audio}/$(basename ${_dir})/${dset}/feats_type"
                        echo "${audio_format}" > "${data_audio}/$(basename ${_dir})/${dset}/audio_format"
                    fi
                done
            done
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    if "${use_speech}"; then
        log "Stage 3a: Perform Kmeans using ${kmeans_feature_type} features"

        if ! "${learn_kmeans}"; then
            kmeans_opts+="--skip_stages 2"
        fi
        for _dir in "data/${dset}/speech/"*; do
            if [ -d "${_dir}" ]; then


                scripts/feats/perform_kmeans.sh \
                    --stage 1 --stop-stage 4 \
                    --train_set "${train_set}" \
                    --dev_set "${valid_set}" \
                    --other_sets "${test_sets}" \
                    --datadir "${data_audio}/$(basename ${_dir})" \
                    --featdir "${data_extract}/$(basename ${_dir})" \
                    --audio_format "${audio_format}" \
                    --feature_type "${kmeans_feature_type}" \
                    --layer "${layer}" \
                    --feature_conf "${kmeans_feature_conf}" \
                    --km_dir "${km_dir}" \
                    --portion "${portion}" \
                    --nclusters "${nclusters}" \
                    --storage_save_mode ${storage_save_mode} \
                    --use_gpu true \
                    --nj ${nj} \
                    --cpu_cmd "${train_cmd}" \
                    --cuda_cmd "${cuda_cmd}" \
                    ${kmeans_opts}
            fi
        done

        _suf=
        if [ -n "${layer}" ]; then
            _suf="layer${layer}/"
        fi

        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            for _dir in "data/${dset}/speech/"*; do
                if [ -d "${_dir}" ]; then
                    utils/copy_data_dir.sh "${data_audio}/$(basename ${_dir})/${dset}" "${data_feats}/${dset}/speech/$(basename ${_dir})"
                    cat "${data_extract}/$(basename ${_dir})/${kmeans_feature_type}/${_suf}${dset}/pseudo_labels_km${nclusters}.txt" \
                        > "${data_feats}/${dset}/speech/$(basename ${_dir})/token"
                fi
            done
        done
    fi

    if "${use_text}"; then
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            for _dir in "data/${dset}/text/"*; do
                if [ -d "${_dir}" ]; then
                    echo "${data_feats}/${dset}/text/$(basename ${_dir})"
                    mkdir -p "${data_feats}/${dset}/text/$(basename ${_dir})"
                    cp "${_dir}/text" "${data_feats}/${dset}/text/$(basename ${_dir})/"
                fi
            done
        done
    fi

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4a: Data filtering: ${data_feats}/org -> ${data_feats}"
    # NOTE(kamo): Not applying to test_sets to keep original data
    if "${use_speech}" && "${use_text}"; then
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            echo $dset
            python3 local/prepare_lm_data.py --path ${data_feats}/${dset}
        done
    fi

    # Create testset
    for _dset in ${test_sets}; do
        python3 local/prepare_lm_test.py --test_file "${data_feats}/${_dset}/lm_text" --path "${data_feats}/${_dset}"
    done

    if [ "${token_type}" = bpe ]; then
        # Create bpe_train_text
        python3 local/prepare_bpe_text.py -i "${data_feats}/${train_set}/lm_text" -o ${bpe_train_text}
    fi
    # shellcheck disable=SC2002
    cat  "${data_feats}/${train_set}/lm_text" | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    if [ "${token_type}" = bpe ]; then
        log "Stage 5: Generate token_list from ${bpe_train_text} using BPE"

        mkdir -p "${bpedir}"
        # shellcheck disable=SC2002
        cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt

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
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage=${bpe_char_cover} \
            --input_sentence_size="${bpe_input_sentence_size}" \
            ${_opts_spm}

        {
        echo "${blank}"
        echo "${oov}"
        # Remove <unk>, <s>, </s> from the vocabulary
        <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${sos_eos}"
        } > "${token_list}"

    elif [ "${token_type}" = char ]; then
        log "Stage 5: Generate character level token_list from ${lm_train_text}"

        _opts="--non_linguistic_symbols ${nlsyms_txt}"

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text  \
            --token_type "${token_type}" \
            --input "${data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

    # check -- remove long sentences?

fi

# ========================== Data preparation is done here. ==========================


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
        _opts+="--config ${lm_config} "
    fi

    # 1. Split the key file
    _logdir="${lm_stats_dir}/logdir"
    mkdir -p "${_logdir}"
    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${data_feats}/lm_train.txt wc -l)" "$(<${lm_dev_text} wc -l)")

    key_file="${data_feats}/lm_train.txt"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${lm_dev_text}"
    split_scps=""
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
    mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

    # 3. Submit jobs
    log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.
    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.lm_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}"\
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_data_path_and_name_and_type "${data_feats}/lm_train.txt,text,text" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${lm_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    <"${lm_stats_dir}/train/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/train/text_shape.${token_type}"

    <"${lm_stats_dir}/valid/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/valid/text_shape.${token_type}"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then
    log "Stage 7: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

    _opts=
    if [ -n "${lm_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
        _opts+="--config ${lm_config} "
    fi

    if [ "${num_splits_lm}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
              --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${token_type}" \
              --num_splits "${num_splits_lm}" \
              --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${token_type} "
    fi

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

    log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 7 using this script"
    mkdir -p "${lm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

    log "LM training started... log: '${lm_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${lm_exp})"
    else
        jobname="${lm_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${lm_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${lm_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.lm_train \
            --ngpu "${ngpu}" \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}"\
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --valid_shape_file "${lm_stats_dir}/valid/text_shape.${token_type}" \
            --fold_length "${lm_fold_length}" \
            --resume true \
            --output_dir "${lm_exp}" \
            ${_opts} ${lm_args}

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    if [ -f ${lm_test_text_textlm} ]; then
        log "Stage 8a: Calc perplexity for textlm: ${lm_test_text_textlm}"
        _opts=
        _output_dir="${lm_exp}/perplexity_test_textlm/$(basename ${lm_test_text_textlm})"
        _ngpu=1     # always use a single GPU since the data is usually small
        log "Perplexity calculation started... log: '${_output_dir}/lm_calc_perplexity.log'"
        # shellcheck disable=SC2086
        ${cuda_cmd} --gpu "${_ngpu}" "${lm_exp}"/perplexity_test_textlm/lm_calc_perplexity.log \
            ${python} -m espnet2.bin.lm_calc_perplexity \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${lm_test_text_textlm},text,text" \
                --train_config "${lm_exp}"/config.yaml \
                --model_file "${lm_exp}/${inference_lm}" \
                --output_dir "${_output_dir}" \
                ${_opts}
        log "PPL: ${lm_test_text_textlm}: $(cat ${_output_dir}/ppl)"
    fi

    if [ -f ${lm_test_text_speechlm} ]; then
        log "Stage 8b: Calc perplexity for unitlm: ${lm_test_text_speechlm}"
        _opts=
        _output_dir="${lm_exp}/perplexity_test_unitlm/$(basename ${lm_test_text_speechlm})"
        _ngpu=1
        log "Perplexity calculation started... log: '${_output_dir}/lm_calc_perplexity.log'"
        # shellcheck disable=SC2086
        ${cuda_cmd} --gpu "${_ngpu}" "${lm_exp}"/perplexity_test_unitlm/lm_calc_perplexity.log \
            ${python} -m espnet2.bin.lm_calc_perplexity \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${lm_test_text_speechlm},text,text" \
                --train_config "${lm_exp}"/config.yaml \
                --model_file "${lm_exp}/${inference_lm}" \
                --output_dir "${_output_dir}" \
                ${_opts}
        log "PPL: ${lm_test_text_speechlm}: $(cat ${_output_dir}/ppl)"
    fi
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    if [ -f ${lm_test_text_asr} ]; then
        log "Stage 9: LM decoding for ASR: ${lm_test_text_asr}"
        _dir="${lm_exp}/decode_test_asr/$(basename ${lm_inference_asr_config} .yaml)"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${lm_inference_asr_config}" ]; then
            _opts+="--config ${lm_inference_asr_config} "
        fi

        # 1. Split the key file
        key_file=${lm_test_text_asr}
        split_scps=""
        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/lm_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/lm_inference.JOB.log \
            ${python} -m espnet2.bin.lm_inference \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${lm_test_text_asr},text,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --output_dir "${_logdir}"/output.JOB \
                --token_type "${token_type}" \
                --bpemodel "${bpemodel}" \
                --lm_train_config "${lm_exp}"/config.yaml \
                --lm_file "${lm_exp}"/${inference_lm}  \
                ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/lm_inference.*.log) ; exit 1; }
            # --log_level "DEBUG"
        # 3. Concatenate output files from each job
        # shellcheck disable=SC2068
        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done

        # 4. Postprocess and score
        _scoredir="${_dir}/score_wer"
        mkdir -p "${_scoredir}"

        python3 local/postprocess.py \
            --input ${_dir}/text \
            --output ${_scoredir}/hyp.trn \
            --sos "<generatetext>" \
            --prefix "asr_"

        python3 local/postprocess.py \
           --input ${data_feats}/test/lm_text \
           --output ${_scoredir}/ref.trn \
           --sos "<generatetext>" \
           --prefix "asr_"

        sclite -r ${_scoredir}/ref.trn trn \
            -h ${_scoredir}/hyp.trn trn \
            -i rm -o all stdout > ${_scoredir}/result.txt

    fi

fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    if [ -f ${lm_test_text_tts} ]; then
        log "Stage 10: LM decoding for TTS: ${lm_test_text_tts}"
        _dir="${lm_exp}/decode_test_tts/$(basename ${lm_test_text_tts})"

        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${lm_inference_tts_config}" ]; then
            _opts+="--config ${lm_inference_tts_config} "
        fi

        # 1. Split the key file
        key_file=${lm_test_text_tts}
        split_scps=""
        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/lm_inference.*.log'"
        rm -f "${_logdir}/*.log"
        # shellcheck disable=SC2046,SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/lm_inference.JOB.log \
            ${python} -m espnet2.bin.lm_inference \
                --batch_size ${batch_size} \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${lm_test_text_tts},text,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --output_dir "${_logdir}"/output.JOB \
                --token_type "${token_type}" \
                --bpemodel "${bpemodel}" \
                --lm_train_config "${lm_exp}"/config.yaml \
                --lm_file "${lm_exp}"/${inference_lm} \
                ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/lm_inference.*.log) ; exit 1; }

        # 3. Concatenate output files from each job
        # shellcheck disable=SC2068
        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done

        # 4. Postprocess
        _scoredir="${_dir}/score_tts"
        mkdir -p "${_scoredir}"

        python3 local/postprocess.py \
            --input ${_dir}/text \
            --output ${_scoredir}/hyp.trn \
            --sos "<generatespeech>" \
            --prefix "tts_"

        # Generate tokens for speech generation
        python3 local/postprocess_speech.py \
            --input ${_scoredir}/hyp.trn \
            --output ${_scoredir}/hyp.tok \
            --prefix "tts_"

    fi

fi


packed_model="${lm_exp}/${lm_exp##*/}_${inference_lm%.*}.zip"
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
    log "Stage 11: Pack model: ${packed_model}"

    _opts=
    if [ "${token_type}" = bpe ]; then
        _opts+="--option ${bpemodel} "
    fi
    if [ "${nlsyms_txt}" != none ]; then
        _opts+="--option ${nlsyms_txt} "
    fi
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack asr \
        --lm_train_config "${lm_exp}"/config.yaml \
        --lm_file "${lm_exp}"/"${inference_lm}" \
        ${_opts} \
        --option "${lm_exp}"/images \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] && ! [[ " ${skip_stages} " =~ [[:space:]]12[[:space:]] ]]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1
    log "Stage 12: Upload model to HuggingFace: ${hf_repo}"

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
    # foo/lm1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=automatic-speech-recognition
    # shellcheck disable=SC2034
    espnet_task=LM
    # shellcheck disable=SC2034
    task_exp=${lm_exp}
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

log "Successfully finished. [elapsed=${SECONDS}s]"
