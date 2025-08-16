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
stage=1                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_data_prep=false    # Skip data preparation stages.
skip_train=false        # Skip training stages.
skip_eval=false         # Skip decoding and evaluation stages.
skip_packing=true       # Skip the packing stage.
skip_upload_hf=true     # Skip uploading to huggingface stage.
ngpu=1                  # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1             # The number of nodes.
nj=32                   # The number of parallel jobs.
inference_nj=32         # The number of parallel jobs in decoding.
gpu_inference=false     # Whether to perform gpu decoding.
dumpdir=dump            # Directory to dump features.
expdir=exp              # Directory to save experiments.
python=python3          # Specify python to execute espnet commands.

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
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3

# Language model related
use_lm=true       # Use language model for SER decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the directory path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the directory path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# SER model related
ser_tag=       # Suffix to the result dir for ser model training.
ser_exp=       # Specify the directory path for SER experiment.
               # If this option is specified, ser_tag is ignored.
ser_stats_dir= # Specify the directory path for SER statistics.
ser_config=    # Config for ser model training.
ser_args=      # Arguments for ser model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in ser config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_ser=1           # Number of splitting for lm corpus.

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_ser_model=valid.acc.ave.pth # SER model path for decoding.
                                      # e.g.
                                      # inference_ser_model=train.loss.best.pth
                                      # inference_ser_model=3epoch.pth
                                      # inference_ser_model=valid.acc.best.pth
                                      # inference_ser_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
use_transcript=false # Use transcripts to incorporate sematic information
pre_postencoder_norm=false
bpe_train_transcript=  # transcript file path of bpe training set.
lm_train_transcript=   # transcript file path of language model training set.
lm_dev_transcript=     # transcript file path of language model development set.
lm_test_transcript=    # transcript file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
ser_speech_fold_length=800 # fold_length for speech data during SER training.
ser_text_fold_length=150   # fold_length for text data during SER training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"
Options:
    # General configuration
    --stage              # Processes starts from the specified stage (default="${stage}").
    --stop_stage         # Processes is stopped at the specified stage (default="${stop_stage}").
    --use_transcript     # Processes is stopped at the specified stage (default="${use_transcript}").
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
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").
    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").
    # SER model related
    --ser_tag          # Suffix to the result dir for ser model training (default="${ser_tag}").
    --ser_exp          # Specify the directory path for SER experiment.
                       # If this option is specified, ser_tag is ignored (default="${ser_exp}").
    --ser_stats_dir    # Specify the directory path for SER statistics (default="${ser_stats_dir}").
    --ser_config       # Config for ser model training (default="${ser_config}").
    --ser_args         # Arguments for ser model training (default="${ser_args}").
                       # e.g., --ser_args "--max_epoch 10"
                       # Note that it will overwrite args in ser config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_ser   # Number of splitting for lm corpus  (default="${num_splits_ser}").
    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_ser_model # SER model path for decoding (default="${inference_ser_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --ser_speech_fold_length # fold_length for speech data during SER training (default="${ser_speech_fold_length}").
    --ser_text_fold_length   # fold_length for text data during SER training (default="${ser_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
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

export LD_PRELOAD=$(g++ -print-file-name=libstdc++.so.6)

# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

if [ $use_transcript = true ]; then
    utt_extra_files="transcript"
else
    utt_extra_files=""
fi

# Use the same text as SER for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as SER for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as SER for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"

# Use the same transcript as SER for bpe training if not specified.
[ -z "${bpe_train_transcript}" ] && bpe_train_transcript="${data_feats}/${train_set}/transcript"
# Use the same transcript as SER for lm training if not specified.
[ -z "${lm_train_transcript}" ] && lm_train_transcript="${data_feats}/${train_set}/transcript"
# Use the same transcript as SER for lm training if not specified.
[ -z "${lm_dev_transcript}" ] && lm_dev_transcript="${data_feats}/${valid_set}/transcript"
# Use the transcript of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_transcript}" ] && lm_test_transcript="${data_feats}/${test_sets%% *}/transcript"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
    transcript_token_listdir=data/${lang}_transcript_token_list
else
    token_listdir=data/token_list
    transcript_token_listdir=data/transcript_token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
transcript_bpedir="${transcript_token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
transcript_bpetoken_list="${transcript_bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
transcript_chartoken_list="${transcript_token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt
transcript_wordtoken_list="${transcript_token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
    transcript_token_list="${transcript_bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    transcript_token_list="${transcript_chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    transcript_token_list="${transcript_wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${ser_tag}" ]; then
    if [ -n "${ser_config}" ]; then
        ser_tag="$(basename "${ser_config}" .yaml)_${feats_type}"
    else
        ser_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        ser_tag+="_${lang}_${token_type}"
    else
        ser_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        ser_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${ser_args}" ]; then
        ser_tag+="$(echo "${ser_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        ser_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    if [ "${lang}" != noinfo ]; then
        lm_tag+="_${lang}_${lm_token_type}"
    else
        lm_tag+="_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${ser_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        ser_stats_dir="${expdir}/ser_stats_${feats_type}_${lang}_${token_type}"
    else
        ser_stats_dir="${expdir}/ser_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        ser_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        ser_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${lang}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${ser_exp}" ]; then
    ser_exp="${expdir}/ser_${ser_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
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
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_ser_model_$(echo "${inference_ser_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
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
        if [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
           for factor in ${speed_perturb_factors}; do
               if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                   scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" \
                        "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           utils/combine_data.sh --extra_files "${utt_extra_files}" "data/${train_set}_sp" ${_dirs}
           for extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py data/"${train_set}_sp"/${extra_file} > data/"${train_set}_sp"/${extra_file}.tmp
                mv data/"${train_set}_sp"/${extra_file}.tmp data/"${train_set}_sp"/${extra_file}
           done
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

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # utils/copy_data_dir.sh --validate_opts --non-print --utt-prefix "MSP-PODCAST_" data/"${dset}" "${data_feats}${_suf}/${dset}"
                mkdir -p "${data_feats}${_suf}/${dset}"
                cp data/"${dset}"/utt2emo "${data_feats}${_suf}/${dset}"
                cp data/"${dset}"/spk2utt "${data_feats}${_suf}/${dset}"
                cp data/"${dset}"/utt2spk "${data_feats}${_suf}/${dset}"
                cp data/"${dset}"/wav.scp "${data_feats}${_suf}/${dset}"
                cp data/"${dset}"/text "${data_feats}${_suf}/${dset}"
                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls data/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                # utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

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

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' \
                    | cut -d, -f2 > ${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}"
            # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                if [ ! -f data/"${dset}"/wav.scp ]; then
		            if [ ! -f data/"${dset}"/segments ]; then
		                <data/"${dset}"/feats.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                    else
		                <data/"${dset}"/segments awk ' { print($2,"<DUMMY>") }' > data/"${dset}"/wav.scp
		            fi
		        fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # Derive the the frame length and feature dimension
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - | \
                    awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

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
            # utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp -r "${data_feats}/org/${dset}" "${data_feats}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            for utt_extra_file in ${utt_extra_files}; do
                cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
            done
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
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(<conf/fbank.conf grep -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(<conf/fbank.conf grep frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                    | awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                <"${data_feats}/org/${dset}/feats.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/feats_shape"  \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"
            for utt_extra_file in ${utt_extra_files}; do
                <"${data_feats}/org/${dset}/${utt_extra_file}" \
                    awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/${dset}/${utt_extra_file}"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            # utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"
            for utt_extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py ${data_feats}/${dset}/${utt_extra_file} \
                    > ${data_feats}/${dset}/${utt_extra_file}.tmp
                mv ${data_feats}/${dset}/${utt_extra_file}.tmp ${data_feats}/${dset}/${utt_extra_file}
            done
        done

        # shellcheck disable=SC2002
        cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
        if "${use_transcript}"; then
            cat ${lm_train_transcript} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train_transcript.txt"
        fi
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = bpe ]; then
            log "Stage 5: Generate token_list from ${bpe_train_text} using BPE"

            mkdir -p "${bpedir}"
            # shellcheck disable=SC2002
            cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt

            if [ -n "${bpe_nlsyms}" ]; then
                _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
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
                --train_extremely_large_corpus=true \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list}"

        elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 5: Generate character level token_list from ${lm_train_text}"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for SLU and also used as ignore-index in the other task
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
        _opts="--non_linguistic_symbols ${nlsyms_txt}"
        if "${use_transcript}"; then
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "word" \
                --input "${data_feats}/lm_train_transcript.txt" --output "${transcript_token_list}" ${_opts} \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"
        fi

        # Create word-list for word-LM training
        if ${use_word_lm} && [ "${token_type}" != word ]; then
            log "Generate word level token_list from ${data_feats}/lm_train.txt"
            ${python} -m espnet2.bin.tokenize_text \
                --token_type word \
                --input "${data_feats}/lm_train.txt" --output "${lm_token_list}" \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --vocabulary_size "${word_vocab_size}" \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"
        fi

    fi
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _ser_train_dir="${data_feats}/${train_set}"
        _ser_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: SER collect stats: train_set=${_ser_train_dir}, valid_set=${_ser_valid_dir}"

        _opts=
        if [ -n "${ser_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.ser_train --print_config --optim adam
            _opts+="--config ${ser_config} "
        fi

        _feats_type="$(<${_ser_train_dir}/feats_type)"
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
            _input_size="$(<${_ser_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${ser_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_ser_train_dir}/${_scp} wc -l)" "$(<${_ser_valid_dir}/${_scp} wc -l)")

        key_file="${_ser_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_ser_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${ser_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${ser_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${ser_stats_dir}/run.sh"; chmod +x "${ser_stats_dir}/run.sh"

        # 3. Submit jobs
        log "SER collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.ser_train \
                --collect_stats true \
                --use_preprocessor true \
                --emotions "A_S_H_U_F_D_C_N_O_X" \
                --train_data_path_and_name_and_type "${_ser_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_ser_train_dir}/utt2emo,emo,text" \
                --valid_data_path_and_name_and_type "${_ser_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_ser_valid_dir}/utt2emo,emo,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${ser_args} || { cat "${_logdir}"/stats.1.log; exit 1; }
        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${ser_stats_dir}"

        # # Append the num-tokens at the last dimensions. This is used for batch-bins count
        # <"${ser_stats_dir}/train/emo_shape" \
        #     awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        #     >"${ser_stats_dir}/train/emo_shape.${token_type}"

        # <"${ser_stats_dir}/valid/emo_shape" \
        #     awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        #     >"${ser_stats_dir}/valid/emo_shape.${token_type}"

    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        _ser_train_dir="${data_feats}/${train_set}"
        _ser_valid_dir="${data_feats}/${valid_set}"
        log "Stage 11: SER Training: train_set=${_ser_train_dir}, valid_set=${_ser_valid_dir}"

        _opts=
        if [ -n "${ser_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.ser_train --print_config --optim adam
            _opts+="--config ${ser_config} "
        fi

        _feats_type="$(<${_ser_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((ser_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${ser_speech_fold_length}"
            _input_size="$(<${_ser_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${ser_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_ser}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${ser_stats_dir}/splits${num_splits_ser}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_ser_train_dir}/${_scp}" \
                      "${_ser_train_dir}/utt2emo" \
                      "${ser_stats_dir}/train/speech_shape" \
                      "${ser_stats_dir}/train/emotion_labels_shape" \
                  --num_splits "${num_splits_ser}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/utt2emo,emo,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/emotion_labels_shape "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_ser_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_ser_train_dir}/utt2emo,emo,text "
            _opts+="--train_shape_file ${ser_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${ser_stats_dir}/train/emotion_labels_shape "
        fi

        log "Generate '${ser_exp}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${ser_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${ser_exp}/run.sh"; chmod +x "${ser_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "SER training started... log: '${ser_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${ser_exp})"
        else
            jobname="${ser_exp}/train.log"
        fi

        if "${use_transcript}"; then
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${ser_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${ser_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.ser_train \
                    --use_preprocessor true \
                    --pre_postencoder_norm "${pre_postencoder_norm}" \
                    --emotions "A_S_H_U_F_D_C_N_O_X" \
                    --valid_data_path_and_name_and_type "${_ser_valid_dir}/${_scp},speech,${_type}" \
                    --valid_data_path_and_name_and_type "${_ser_valid_dir}/utt2emo,emo,text" \
                    --valid_shape_file "${ser_stats_dir}/valid/speech_shape" \
                    --valid_shape_file "${ser_stats_dir}/valid/emotion_labels_shape" \
                    --resume true \
                    --init_param ${pretrained_model} \
                    --ignore_init_mismatch ${ignore_init_mismatch} \
                    --fold_length "${_fold_length}" \
                    --fold_length "${ser_text_fold_length}" \
                    --fold_length "${ser_text_fold_length}" \
                    --output_dir "${ser_exp}" \
                    ${_opts} ${ser_args}
        else
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${ser_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${ser_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.ser_train \
                    --use_preprocessor true \
                    --emotions "A_S_H_U_F_D_C_N_O_X" \
                    --valid_data_path_and_name_and_type "${_ser_valid_dir}/${_scp},speech,${_type}" \
                    --valid_data_path_and_name_and_type "${_ser_valid_dir}/utt2emo,emo,text" \
                    --valid_shape_file "${ser_stats_dir}/valid/speech_shape" \
                    --valid_shape_file "${ser_stats_dir}/valid/emotion_labels_shape" \
                    --resume true \
                    --init_param ${pretrained_model} \
                    --ignore_init_mismatch ${ignore_init_mismatch} \
                    --fold_length "${_fold_length}" \
                    --fold_length "${ser_text_fold_length}" \
                    --output_dir "${ser_exp}" \
                    ${_opts} ${ser_args}
        fi

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    ser_exp="${expdir}/${download_model}"
    mkdir -p "${ser_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${ser_exp}/config.txt"

    # Get the path of each file
    _ser_model_file=$(<"${ser_exp}/config.txt" sed -e "s/.*'ser_model_file': '\([^']*\)'.*$/\1/")
    _ser_train_config=$(<"${ser_exp}/config.txt" sed -e "s/.*'ser_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_ser_model_file}" "${ser_exp}"
    ln -sf "${_ser_train_config}" "${ser_exp}"
    inference_ser_model=$(basename "${_ser_model_file}")

    if [ "$(<${ser_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${ser_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${ser_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Predicting: training_dir=${ser_exp}"

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

        # 2. Generate run.sh
        log "Generate '${ser_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${ser_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${ser_exp}/${inference_tag}/run.sh"; chmod +x "${ser_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${ser_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""

            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            ser_inference_tool="espnet2.bin.ser_inference"

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/ser_inference.*.log'"
            # shellcheck disable=SC2086
            if "${use_transcript}"; then
                ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/ser_inference.JOB.log \
                    ${python} -m ${ser_inference_tool} \
                        --batch_size ${batch_size} \
                        --ngpu "${_ngpu}" \
                        --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                        --data_path_and_name_and_type "${_data}/transcript,transcript,text" \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --ser_train_config "${ser_exp}"/config.yaml \
                        --ser_model_file "${ser_exp}"/"${inference_ser_model}" \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_opts} ${inference_args}
            else
                ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/ser_inference.JOB.log \
                ${python} -m ${ser_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --ser_train_config "${ser_exp}"/config.yaml \
                    --ser_model_file "${ser_exp}"/"${inference_ser_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}
            fi

            # 3. Concatenates the output files from each jobs
            for f in emo; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    fi

    # fi
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 13: SER Scoring"
        # Define emotion-to-index mapping
        emotions=(A S H U F D C N O X)

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${ser_exp}/${inference_tag}/${dset}"
            _utt2emo="data/${dset}/utt2emo"
            _reffile="${_dir}/ref"

            if [ ! -f "${_utt2emo}" ]; then
                log "Missing utt2emo: ${_utt2emo}"
                exit 1
            fi

            # Create ref file by converting emotions to indices
            paste "${_utt2emo}" <(
                while read -r line; do
                    emo=${line#* }  # Get second field
                    for i in "${!emotions[@]}"; do
                        if [ "${emotions[$i]}" = "$emo" ]; then
                            echo $i
                            break
                        fi
                    done
                done <"${_utt2emo}"
            ) | awk '{print $1, $3}' > "${_reffile}"

            log "Created reference label file at: ${_reffile}"
        done

        for dset in ${test_sets}; do
            _dir="${ser_exp}/${inference_tag}/${dset}"
            _scoredir="${_dir}/score"
            mkdir -p "${_scoredir}"

            # Call scoring script
            ${python} pyscripts/utils/score_ser.py \
                --ref "${_dir}/ref" \
                --hyp "${_dir}/emo" \
                | tee "${_scoredir}/metrics.txt"

            log "SER evaluation results saved to ${_scoredir}/metrics.txt"
        done

        python3 scripts/utils/show_ser_result.py \
                --metrics "${_scoredir}/metrics.txt" \
                --dataset "${inference_tag}/${test_sets}" > "${ser_exp}"/RESULTS.md
        cat "${ser_exp}"/RESULTS.md
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${ser_exp}/${ser_exp##*/}_${inference_ser_model%.*}.zip"
if ! "${skip_packing}" && [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model or skip_packing is true
    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Pack model: ${packed_model}"

        _opts=
        ${python} -m espnet2.bin.pack ser \
            --ser_train_config "${ser_exp}"/config.yaml \
            --ser_model_file "${ser_exp}"/"${inference_ser_model}" \
            ${_opts} \
            --option "${ser_exp}"/RESULTS.md \
            --option "${ser_exp}"/RESULTS.md \
            --option "${ser_exp}"/images \
            --outpath "${packed_model}"
    fi
else
    log "Skip the packing stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 15: Upload model to HuggingFace: ${hf_repo}"

        if [ ! -f "${packed_model}" ]; then
            log "ERROR: ${packed_model} does not exist. Please run stage 14 first."
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
        hf_task=speech-emotion-recognition
        # shellcheck disable=SC2034
        espnet_task=SER
        # shellcheck disable=SC2034
        task_exp=${ser_exp}
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
