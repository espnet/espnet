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
skip_stages=         # Spicify the stage to be skipped
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading to zenodo.
skip_upload_hf=true  # Skip uploading to hugging face stages.
eval_valid_set=false # Run decoding for the validation set
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
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
feats_type=raw       # Feature type (raw).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.

# Kmeans related
kmeans_opts=                # The options given to scripts/feats/perform_kmeans.sh
kmeans_feature="wavlm_large/21" # format: ssl_model_type/layer_idx (e.g. mfcc, hubert_large/21, wavlm_large/21)
portion=0.1
nclusters=2000              # The number of clusters for discrete tokenss
storage_save_mode=true      # Save storage on SSL feature extraction
                            # If true, feature extraction and kmeans clustering on the fly

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
not_avail="<na>"    # not available symbole. Sometimes, we do not have CTC targets for MT task.
token_joint=false       # whether to use a single bpe system for both source and target languages
src_token_type=bpe      # Tokenization type (char or bpe) for source languages.
src_nbpe=30             # The number of BPE vocabulary for source language.
src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe).
src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language.
src_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE of source language
src_bpe_char_cover=1.0  # character coverage when modeling BPE for source language
tgt_token_type=bpe      # Tokenization type (char or bpe) for target language.
tgt_nbpe=30             # The number of BPE vocabulary for target language.
tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language.
tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language.
tgt_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE for target language.
tgt_bpe_char_cover=1.0  # character coverage when modeling BPE for target language.

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3

# Language model related
use_lm=true       # Use language model for ST decoding.
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

# ST model related
st_tag=        # Suffix to the result dir for st model training.
st_exp=        # Specify the directory path for ST experiment.
                # If this option is specified, st_tag is ignored.
st_stats_dir=  # Specify the directory path for ST statistics.
st_config=     # Config for st model training.
st_args=       # Arguments for st model training, e.g., "--max_epoch 10".
                # Note that it will overwrite args in st config.
ignore_init_mismatch=false      # Ignore initial mismatch
num_splits_st=1                 # Number of splitting for lm corpus.
speech_token_case="ts"          # speech discrete token  case. ts: true sequence, rm: remove duplicated tokens
speech_token_lang="wavlm_large_21_km2000"  # speech discrete token type abbrev. id (e.g., wavlm_large_21_km2000)
src_tgt_text_case="lc.rm"       # source / target transcript case. Note, all source / target text should use the same case for now.
src_tgt_text_lang=en            # source / target language abbrev. id (e.g., en). Multiple langs are supported to support multiple tasks, with space between (e.g., "es/en"), from data's perspect of view, src_lang of text is the first.multiple tasks, with space between (e.g., "es/en"), from data's perspect of view, src_lang of text is the first.
tgt_tasks="asr/st"              # task abbrev. id (e.g., st). Multiple tasks are supported to support multiple tasks, with space between (e.g., "asr/st")

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
                         # and transformer language model for nbest rescoring

batch_size=1
inference_asr_tag=    # Suffix to the result dir for asr decoding.
inference_st_tag=    # Suffix to the result dir for st decoding.
inference_mt_tag=    # Suffix to the result dir for mt decoding.
inference_asr_config= # Config for asr decoding.
inference_st_config=  # Config for st decoding.
inference_mt_config=  # Config for mt decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_st_model=valid.acc.ave.pth  # ST model path for decoding.
                                      # e.g.
                                      # inference_st_model=train.loss.best.pth
                                      # inference_st_model=3epoch.pth
                                      # inference_st_model=valid.acc.best.pth
                                      # inference_st_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
src_bpe_train_text=  # Text file path of bpe training set for source language.
tgt_bpe_train_text=  # Text file path of bpe training set for target language.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
st_text_fold_length=150   # fold_length for text data during ST training.
lm_fold_length=150         # fold_length for LM training.
min_speech_token_len=4     # Minimum duration in second.
max_speech_token_len=1500  # Maximum duration in second.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_stages    # Spicify the stage to be skipped (default="${skip_stages}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --skip_upload_hf    # Skip packing and uploading stages (default="${skip_upload_hf}").
    --eval_valid_set # Run decoding for the validation set (default="${eval_valid_set}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw or raw_copy, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").

    # Kmeans related
    --kmeans_opts       # The options given to kmeans step (default="${kmeans_opts}").
    --kmeans_feature    # The string indicates the kmeans features (default="${kmeans_feature}").
    --portion           # The portion of data used to train kmeans (default="${portion}").
    --nclusters         # The number of clusters for discrete tokens (default="${nclusters}").
    --storage_save_mode # # Save storage on SSL feature extraction. If true, feature extraction and kmeans clustering on the fly (default="${storage_save_mode}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --not_avail               # not available symbole. Sometimes, we do not have CTC targets for MT task. (default="${not_avail}").
    --token_joint=false       # Whether to use a single bpe system for both source and target languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --src_token_type          # Tokenization type (char or bpe) for source languages. (default="${src_token_type}").
    --src_nbpe                # The number of BPE vocabulary for source language. (default="${src_nbpe}").
    --src_bpemode             # Mode of BPE for source language (unigram or bpe). (default="${src_bpemode}").
    --src_bpe_input_sentence_size  # Size of input sentence for BPE for source language. (default="${src_bpe_input_sentence_size}").
    --src_bpe_nlsyms          # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${src_bpe_nlsyms}").
    --src_bpe_char_cover      # Character coverage when modeling BPE for source language. (default="${src_bpe_char_cover}").
    --tgt_token_type          # Tokenization type (char or bpe) for target language. (default="${tgt_token_type}").
    --tgt_nbpe                # The number of BPE vocabulary for target language. (default="${tgt_nbpe}").
    --tgt_bpemode             # Mode of BPE (unigram or bpe) for target language. (default="${tgt_bpemode}").
    --tgt_bpe_input_sentence_size  # Size of input sentence for BPE for target language. (default="${tgt_bpe_input_sentence_size}").
    --tgt_bpe_nlsyms          # Non-linguistic symbols list, separated by a comma, for BPE for target language. (default="${tgt_bpe_nlsyms}").
    --tgt_bpe_char_cover      # Character coverage when modeling BPE for target language. (default="${tgt_bpe_char_cover}").

    # Language model related
    --lm_tag            # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp            # Specify the directory path for LM experiment.
                        # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir      # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config         # Config for language model training (default="${lm_config}").
    --lm_args           # Arguments for language model training (default="${lm_args}").
                        # e.g., --lm_args "--max_epoch 10"
                        # Note that it will overwrite args in lm config.
    --use_word_lm       # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size   # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm     # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ST model related
    --st_tag            # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp            # Specify the directory path for ST experiment.
                        # If this option is specified, st_tag is ignored (default="${st_exp}").
    --st_stats_dir      # Specify the directory path for ST statistics (default="${st_stats_dir}").
    --st_config         # Config for st model training (default="${st_config}").
    --st_args           # Arguments for st model training (default="${st_args}").
                        # e.g., --st_args "--max_epoch 10"
                        # Note that it will overwrite args in st config.
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --num_splits_st       # Number of splitting for lm corpus.  (default="${num_splits_st}").
    --speech_token_case   # source case abbrev. id (e.g., wavlm_large_21_km2000). (default="${speech_token_case}")
    --speech_token_lang   # source language abbrev. id (e.g., es). (default="${speech_token_lang}")
    --src_tgt_text_case   # target case abbrev. id (e.g., lc.rm). (default="${src_tgt_text_case}")
    --src_tgt_text_lang   # target language abbrev. id (e.g., en). Multiple langs are supported for multiple tasks (default="${src_tgt_text_lang}"), from data's perspect of view, src_lang of text is the first.
    --tgt_tasks           # target task abbrev. id (e.g., st). Multiple tasks are supported (default="${tgt_tasks}")

    # Decoding related
    --inference_asr_tag     # Suffix to the result dir for asr decoding (default="${inference_asr_tag}").
    --inference_st_tag      # Suffix to the result dir for st decoding (default="${inference_st_tag}").
    --inference_mt_tag      # Suffix to the result dir for mt decoding (default="${inference_mt_tag}").
    --inference_asr_config  # Config for asr decoding (default="${inference_asr_config}").
    --inference_st_config   # Config for st decoding (default="${inference_st_config}").
    --inference_mt_config   # Config for mt decoding (default="${inference_mt_config}").
    --inference_args        # Arguments for decoding (default="${inference_args}").
                            # e.g., --inference_args "--lm_weight 0.1"
                            # Note that it will overwrite args in inference config.
    --inference_lm          # Language model path for decoding (default="${inference_lm}").
    --inference_st_model    # ST model path for decoding (default="${inference_st_model}").
    --download_model        # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set             # Name of training set (required).
    --valid_set             # Name of validation set used for monitoring/tuning network training (required).
    --test_sets             # Names of test sets.
                            # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --src_bpe_train_text    # Text file path of bpe training set for source language.
    --tgt_bpe_train_text    # Text file path of bpe training set for target language
    --lm_train_text         # Text file path of language model training set.
    --lm_dev_text           # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text          # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt            # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner               # Text cleaner (default="${cleaner}").
    --g2p                   # g2p method (default="${g2p}").
    --lang                  # The language type of corpus (default=${lang}).
    --score_opts            # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts      # The options given to local/score.sh (default="{local_score_opts}").
    --st_text_fold_length   # fold_length for text data during ST training (default="${st_text_fold_length}").
    --lm_fold_length        # fold_length for LM training (default="${lm_fold_length}").
    --min_speech_token_len   # minimum length for speech token text, (default="${min_speech_token_len}").
    --max_speech_token_len   # maximum length for speech token text, (default="${max_speech_token_len}").
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
if ! "${skip_train}"; then
    [ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
    [ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
fi
if ! "${eval_valid_set}"; then
    [ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };
else
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

# Extra files for translation process
utt_extra_files="utt2category text.${speech_token_case}.${speech_token_lang} text.${src_tgt_text_case}.${src_tgt_text_lang//\//_} text.${src_tgt_text_case}.ctc"
utt_extra_files_names="utt2category src_text text text_ctc"
utt_extra_files=(${utt_extra_files// / })
utt_extra_files_names=(${utt_extra_files_names// / })
[ "${#utt_extra_files[@]}" -eq "${#utt_extra_files_names[@]}" ] || exit 1;
src_tgt_lang_lst=(${src_tgt_text_lang//\// })
if [ "${#src_tgt_lang_lst[@]}" -ne 2 ]; then
    log "Only 2 languages are supported due to data preparation. ${#src_tgt_lang_lst[@]} were detected: ${src_tgt_text_lang}" && exit 1;
fi
ref_text_files=
for l in ${src_tgt_lang_lst[@]}; do
    ref_text_files+="text.${src_tgt_text_case}.${l} "
done

# Use the same text as ST for bpe training if not specified.
[ -z "${src_bpe_train_text}" ] && src_bpe_train_text="${data_feats}/${train_set}.${tgt_tasks//\//_}/text.${speech_token_case}.${speech_token_lang}"
[ -z "${tgt_bpe_train_text}" ] && tgt_bpe_train_text="${data_feats}/${train_set}.${tgt_tasks//\//_}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_} ${data_feats}/${train_set}.${tgt_tasks//\//_}/text.${src_tgt_text_case}.ctc"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}.${tgt_tasks//\//_}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}.${tgt_tasks//\//_}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}.${tgt_tasks//\//_}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}"

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
# The tgt bpedir is set for all cases when using bpe
if "${token_joint}"; then
    tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}.${tgt_tasks//\//_}"
else
    tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}_${src_tgt_text_case}_${src_tgt_text_lang//\//_}.${tgt_tasks//\//_}"
fi
tgt_bpeprefix="${tgt_bpedir}"/bpe
tgt_bpemodel="${tgt_bpeprefix}".model
tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
tgt_chartoken_list="${token_listdir}"/char_${src_tgt_text_lang//\//_}/tgt_tokens.txt
if "${token_joint}"; then
    # if token_joint, the bpe training will use both speech_token_lang and src_tgt_text_lang to train a single bpe model
    src_bpedir="${tgt_bpedir}"
    src_bpeprefix="${tgt_bpeprefix}"
    src_bpemodel="${tgt_bpemodel}"
    src_bpetoken_list="${tgt_bpetoken_list}"
    src_chartoken_list="${tgt_chartoken_list}"
else
    src_bpedir="${token_listdir}/src_bpe_${src_bpemode}${src_nbpe}_${speech_token_case}_${speech_token_lang}.${tgt_tasks//\//_}"
    src_bpeprefix="${src_bpedir}"/bpe
    src_bpemodel="${src_bpeprefix}".model
    src_bpetoken_list="${src_bpedir}"/tokens.txt
    src_chartoken_list="${token_listdir}"/char_${speech_token_lang}/src_tokens.txt
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
tgt_wordtoken_list="${token_listdir}"/word_${src_tgt_text_lang//\//_}/tgt_tokens.txt
if "${token_joint}"; then
    src_wordtoken_list="${tgt_wordtoken_list}"
else
    src_wordtoken_list="${token_listdir}"/word_${speech_token_lang}/src_tokens.txt
fi

# Set token types for src and tgt langs
if [ "${src_token_type}" = bpe ]; then
    src_token_list="${src_bpetoken_list}"
elif [ "${src_token_type}" = char ]; then
    src_token_list="${src_chartoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = word ]; then
    src_token_list="${src_wordtoken_list}"
    src_bpemodel=none
else
    log "Error: not supported --src_token_type '${src_token_type}'"
    exit 2
fi
if [ "${tgt_token_type}" = bpe ]; then
    tgt_token_list="${tgt_bpetoken_list}"
elif [ "${tgt_token_type}" = char ]; then
    tgt_token_list="${tgt_chartoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = word ]; then
    tgt_token_list="${tgt_wordtoken_list}"
    tgt_bpemodel=none
else
    log "Error: not supported --tgt_token_type '${tgt_token_type}'"
    exit 2
fi
# NOTE: keep for future development.
# shellcheck disable=SC2317
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${tgt_wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${tgt_token_list}"
    lm_token_type="${tgt_token_type}"
fi

if [ ${kmeans_feature} = "mfcc" ]; then  # MFCC has no layer
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=
    kmeans_feature_conf="{type=mfcc}"
else
    kmeans_feature_type=$(echo "${kmeans_feature}" | cut -d/ -f1)
    layer=$(echo "${kmeans_feature}" | cut -d/ -f2)
    # TODO(simpleoier): to support features beyond s3prl
    s3prl_conf="{upstream=${kmeans_feature_type}}"
    kmeans_feature_conf="{type=s3prl,conf={s3prl_conf=${s3prl_conf},download_dir=ckpt,multilayer_feature=False,layer=${layer}}}"
fi
km_dir="${expdir}"/kmeans/$(echo "${kmeans_feature}" | tr "/" "_")_${nclusters}clusters

# Set tag for naming of model directory
if [ -z "${st_tag}" ]; then
    if [ -n "${st_config}" ]; then
        st_tag="$(basename "${st_config}" .yaml)_${feats_type}"
    else
        st_tag="train_${feats_type}"
    fi
    if [ "${speech_token_lang}" != noinfo ]; then
        st_tag+="_${speech_token_lang}_${src_token_type}_${speech_token_case}"
    else
        st_tag+="_${src_token_type}_${speech_token_case}"
    fi
    if ${token_joint}; then
        st_tag+="_joint"
    elif [ "${src_token_type}" = bpe ]; then
        st_tag+="${src_nbpe}"
    fi
    if [ "${src_tgt_text_lang//\//_}" != noinfo ]; then
        st_tag+="_${src_tgt_text_lang//\//_}_${tgt_token_type}_${src_tgt_text_case}"
    else
        st_tag+="_${tgt_token_type}_${src_tgt_text_case}"
    fi
    if [ "${tgt_token_type}" = bpe ]; then
        st_tag+="${tgt_nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${st_args}" ]; then
        st_tag+="$(echo "${st_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_tag+="_sp"
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
        lm_tag+="${tgt_nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${st_stats_dir}" ]; then
    if [ "${speech_token_lang}" != noinfo ]; then
        st_stats_dir="${expdir}/${tgt_tasks//\//_}_stats_${feats_type}_${speech_token_case}_${speech_token_lang}_${src_token_type}"
    else
        st_stats_dir="${expdir}/${tgt_tasks//\//_}_stats_${feats_type}_${speech_token_case}_${src_token_type}"
    fi
    if ${token_joint}; then
        st_stats_dir+="_joint"
    elif [ "${src_token_type}" = bpe ]; then
        st_stats_dir+="${src_nbpe}"
    fi
    if [ "${src_tgt_text_lang//\//_}" != noinfo ]; then
        st_stats_dir+="_${src_tgt_text_lang//\//_}_${tgt_token_type}"
    else
        st_stats_dir+="_${tgt_token_type}"
    fi
    if [ "${tgt_token_type}" = bpe ]; then
        st_stats_dir+="${tgt_nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    if [ "${src_tgt_text_lang//\//_}" != noinfo ]; then
        lm_stats_dir="${expdir}/lm_stats_${src_tgt_text_lang//\//_}_${lm_token_type}"
    else
        lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${tgt_nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${st_exp}" ]; then
    st_exp="${expdir}/${tgt_tasks//\//_}_${st_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi

if [ -z "${inference_asr_tag}" ]; then
    if [ -n "${inference_asr_config}" ]; then
        inference_asr_tag="$(basename "${inference_asr_config}" .yaml)"
    else
        inference_asr_tag=inference
    fi
    if [ -n "${inference_st_config}" ]; then
        inference_st_tag="$(basename "${inference_st_config}" .yaml)"
    else
        inference_st_tag=inference
    fi
    if [ -n "${inference_mt_config}" ]; then
        inference_mt_tag="$(basename "${inference_mt_config}" .yaml)"
    else
        inference_mt_tag=inference
    fi

    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_asr_tag+="$(echo "${inference_asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
        inference_st_tag+="$(echo "${inference_st_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
        inference_mt_tag+="$(echo "${inference_mt_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_asr_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
        inference_st_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
        inference_mt_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_asr_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
        inference_st_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
        inference_mt_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_asr_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    inference_st_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    inference_mt_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
        inference_asr_tag+="_use_k2"
        inference_st_tag+="_k2_ctc_decoding_${k2_ctc_decoding}"
        inference_mt_tag+="_use_nbest_rescoring_${use_nbest_rescoring}"
    fi
fi

if "${skip_data_prep}"; then
    skip_stages+=" 1 2 3 4 5 6"
fi
if "${skip_train}"; then
    skip_stages+=" 5 6 7 8 9 10 11 12 13"
elif ! "${use_lm}"; then
    skip_stages+=" 8 9 10"
fi
if ! "${use_ngram}"; then
    skip_stages+=" 11"
fi
if "${skip_eval}"; then
    skip_stages+=" 14 15"
fi

if "${skip_upload}" && "${skip_upload_hf}"; then
    skip_stages+=" 16 17 18"
elif "${skip_upload}"; then
    skip_stages+=" 17"
elif "${skip_upload_hf}"; then
    skip_stages+=" 18"
fi
skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"

# ========================== Main stages start from here. ==========================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]] ]]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh --src_lang ${src_tgt_lang_lst[0]} --tgt_lang ${src_tgt_lang_lst[1]} ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
        log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

        for factor in ${speed_perturb_factors}; do
            if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                scripts/utils/perturb_data_dir_speed.sh \
                    ${ref_text_files:+--utt_extra_files "${ref_text_files}"} \
                    "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                _dirs+="data/${train_set}_sp${factor} "
            else
                # If speed factor is 1, same as the original
                _dirs+="data/${train_set} "
            fi
        done
        utils/combine_data.sh \
            ${ref_text_files:+--extra_files "${ref_text_files}"} \
            "data/${train_set}_sp" ${_dirs}
    else
        log "Skip stage 2: Speed perturbation"
    fi
fi

train_sp_sets=
if [ -n "${speed_perturb_factors}" ]; then
    for factor in ${speed_perturb_factors}; do
        if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
            train_sp_sets+="${train_set}_sp${factor} "
        fi
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! [[ " ${skip_stages} " =~ [[:space:]]3[[:space:]] ]]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${train_sp_sets} ${valid_set} ${test_sets}"
    fi
    if [ "${feats_type}" = raw ]; then
        log "Stage 3: Format wav.scp: data/ -> ${data_audio}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_audio}/${dset}"
            rm -f "${data_audio}/${dset}"/{segments,wav.scp,reco2file_and_channel,reco2dur}

            _opts=
            if [ -e "data/${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${data_audio}/${dset}"

            echo "${feats_type}" > "${data_audio}/${dset}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_audio}/${dset}/audio_format"
            else
                echo "${audio_format}" > "${data_audio}/${dset}/audio_format"
            fi
        done
    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! [[ " ${skip_stages} " =~ [[:space:]]4[[:space:]] ]]; then
    log "Stage 4a: Perform Kmeans using ${kmeans_feature_type} features"

    scripts/feats/perform_kmeans.sh \
        --stage 1 --stop-stage 4 \
        --train_set "${train_set}" \
        --dev_set "${valid_set}" \
        --other_sets "${test_sets} ${train_sp_sets}" \
        --datadir "${data_audio}" \
        --featdir "${data_extract}" \
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

    log "Stage 4b: Prepare token_list and convert number indices to CJK tokens"

    # Get uniq chars
    if [ ! -f "${km_dir}/../"distinct_cjk_token_lists ]; then
        if [ ${nclusters} -ge 20900 ]; then
            echo "Warning: too many clusters, be careful with the distinct token list."
        fi
        python3 -c "for i in range(${nclusters}): print(i, chr(int('4e00', 16) + i))" \
            > "${km_dir}/../"distinct_cjk_token_lists
    fi

    _suf=
    if [ -n "${layer}" ]; then
        _suf="layer${layer}/"
    fi

    if [ "${speech_token_case}" = ts ]; then
        echo "keep the original discrete token sequence"
        for dset in "${train_set}" ${train_sp_sets} "${valid_set}" ${test_sets}; do
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=2; i<=NF; i++) {
                        out=out""a[$i];
                    }
                    print($1,out);
                }' "${km_dir}/../"distinct_cjk_token_lists \
                "${data_extract}/${kmeans_feature_type}/${_suf}${dset}/pseudo_labels_km${nclusters}.txt" \
                > "data/${dset}"/text.${speech_token_case}.${speech_token_lang}
        done
    elif [ "${speech_token_case}" = rm ]; then
        echo "remove repetitions in the discrete token sequence"
        for dset in "${train_set}" ${train_sp_sets} "${valid_set}" ${test_sets}; do
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=2; i<=NF; i++) {
                        if ($i != $(i-1)) {out=out""a[$i]}
                    }
                    print($1,out);
                }' "${km_dir}/../"distinct_cjk_token_lists \
                "${data_extract}/${kmeans_feature_type}/${_suf}${dset}/pseudo_labels_km${nclusters}.txt" \
                > "data/${dset}/text.${speech_token_case}.${speech_token_lang}"
        done
    else
        echo "Unrecognized speech_token_case ${speech_token_case}" && exit 1;
    fi

    _src_lang=${src_tgt_lang_lst[0]}
    _tgt_lang=${src_tgt_lang_lst[1]}

    # prepare input and target files for each task
    orig_task_input_files=
    orig_task_label_files=
    label_file_langs=
    for task_id in $(echo ${tgt_tasks} | tr "/" " "); do
        if [ ${task_id} = "asr" ]; then
            orig_task_input_files+="text.${speech_token_case}.${speech_token_lang} "
            orig_task_label_files+="text.${src_tgt_text_case}.${_src_lang} "
            label_file_langs+="${_src_lang} "
        elif [ ${task_id} = "st" ]; then
            orig_task_input_files+="text.${speech_token_case}.${speech_token_lang} "
            orig_task_label_files+="text.${src_tgt_text_case}.${_tgt_lang} "
            label_file_langs+="${_tgt_lang} "
        elif [ ${task_id} = "mt" ]; then
            orig_task_input_files+="text.${src_tgt_text_case}.${_src_lang} "
            orig_task_label_files+="text.${src_tgt_text_case}.${_tgt_lang} "
            label_file_langs+="${_tgt_lang} "
        else
            echo "Unsupported task: ${task_id}" && exit 1;
        fi
    done
    orig_task_input_files=(${orig_task_input_files// / })
    orig_task_label_files=(${orig_task_label_files// / })
    label_file_langs=(${label_file_langs// / })

    tgt_tasks_lst=(${tgt_tasks//\// })

    for dset in "${train_set}" ${train_sp_sets} "${valid_set}"; do  # Only combine the train / valid sets.
        _dest_dir=data/${dset}$(echo "."${tgt_tasks} | tr "/" "_")

        [ -d ${_dest_dir} ] && rm -r ${_dest_dir}
        mkdir -p ${_dest_dir}

        #TODO(simpleoier): mt data length is not specially considered in this stage.
        <${data_extract}/${kmeans_feature_type}/${_suf}${dset}/utt2num_frames_km${nclusters} \
            awk -v tasks="${tgt_tasks}" '
                BEGIN{n_tasks=split(tasks, task_lst, "/");}
                {uttname=$1; for (i=1; i<=n_tasks; i++) {$1=uttname"_"task_lst[i]; print($0)}}' | \
                sort -u > ${_dest_dir}/utt2num_frames.${speech_token_lang}

        # attn-dec text
        for i in ${!orig_task_label_files[@]}; do
            awk -v task=${tgt_tasks_lst[i]} -v lang=${label_file_langs[i]} '{
                $1=$1"_"task; $2="<"task">""<"lang">"$2; print($0)
                }' data/${dset}/${orig_task_label_files[i]}
        done | sort -u > ${_dest_dir}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}
        # ctc text
        <data/${dset}/text.${src_tgt_text_case}.${_src_lang} \
            awk -v tasks="${tgt_tasks}" -v no_ctc_tasks="mt" -v not_avail="${not_avail}" '
                BEGIN{
                    n_tasks=split(tasks, task_lst, "/"); n=split(no_ctc_tasks, lst, "/");
                    for (t=1; t<=n; t++) {no_ctc_tasks_lst[lst[t]]=1}
                }{
                    uttname=$1;
                    for (i=1; i<=n_tasks; i++) {
                        if (task_lst[i] in no_ctc_tasks_lst) {
                            print(uttname"_"task_lst[i], not_avail);
                        } else {
                            $1=uttname"_"task_lst[i]; print($0)
                        }
                    }
                }' | sort -u > ${_dest_dir}/text.${src_tgt_text_case}.ctc
        # input text
        for i in ${!orig_task_input_files[@]}; do
            awk -v task=${tgt_tasks_lst[i]} '{
                $1=$1"_"task; print($0)
                }' data/${dset}/${orig_task_input_files[i]}
        done | sort -u > ${_dest_dir}/text.${speech_token_case}.${speech_token_lang}
        # wav.scp utt2spk
        for f in wav.scp utt2spk; do
            awk -v tasks="${tgt_tasks}" '
                BEGIN{n_tasks=split(tasks, task_lst, "/");}
                {uttname=$1; for (i=1; i<=n_tasks; i++) {$1=uttname"_"task_lst[i]; print($0)}}' \
                data/${dset}/${f} | sort -u > ${_dest_dir}/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${_dest_dir}/utt2spk > ${_dest_dir}/spk2utt
        # utt2category
        awk '{n=split($1, lst, "_"); print($1, lst[n])}' ${_dest_dir}/utt2spk > ${_dest_dir}/utt2category
    done

    if [ -n "${speed_perturb_factors}" ]; then
        _train_set=$(echo "${train_set}.${tgt_tasks}" | tr "/" "_")
        _dirs="data/${_train_set} "
        for factor in ${speed_perturb_factors}; do
            if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                _train_set=$(echo "${train_set}_sp${factor}.${tgt_tasks}" | tr "/" "_")
                _dirs+="data/${_train_set} "
            fi
        done
        _destdset=$(echo "${train_set}_sp.${tgt_tasks}" | tr "/" "_")
        utils/combine_data.sh \
            --extra_files "$(echo ${utt_extra_files[@]}) utt2num_frames.${speech_token_lang} " \
            "data/${_destdset}" ${_dirs}
    fi
fi


if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
fi
if "${eval_valid_set}"; then    # add the original valid set in the test_sets.
    test_sets="${valid_set} ${test_sets}"
fi
train_set=$(echo "${train_set}.${tgt_tasks}" | tr "/" "_")
valid_set=$(echo "${valid_set}.${tgt_tasks}" | tr "/" "_")
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! [[ " ${skip_stages} " =~ [[:space:]]5[[:space:]] ]]; then
    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${train_set} ${valid_set} ${test_sets}"
    fi
    if [ "${feats_type}" = raw ]; then
        log "Stage 5: data/ -> ${data_feats}"

        for dset in ${_dsets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

            if [ ${#utt_extra_files[@]} -ge 1 ]; then
                for extra_file in ${utt_extra_files[@]} "utt2num_frames.${speech_token_lang}"; do
                    # with regex to suuport multi-references
                    [ -f data/${dset}/${extra_file} ] && \
                        cp data/${dset}/${extra_file} ${data_feats}${_suf}/${dset}
                done
            fi
            # Copy reference text files if there is more than 1 reference
            if [[ "${test_sets}" =~ "${dset}" ]] && [ -n "${ref_text_files}" ]; then
                # shellcheck disable=SC2068
                for ref_txt in ${ref_text_files}; do
                    [ -f data/${dset}/${ref_txt} ] && cp data/${dset}/${ref_txt} ${data_feats}${_suf}/${dset}
                done
            fi
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done
    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! [[ " ${skip_stages} " =~ [[:space:]]6[[:space:]] ]]; then
    log "Stage 6: Data filtering: ${data_feats}/org -> ${data_feats}"

    # NOTE(kamo): Not applying to test_sets to keep original data
    for dset in "${train_set}" "${valid_set}"; do
        # Copy data dir
        utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"

        for utt_extra_file in feats_type ${utt_extra_files[@]}; do
            cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
        done

        # Remove short or long input text
        <"${data_feats}/org/${dset}/utt2num_frames.${speech_token_lang}" \
            awk -v min_length="${min_speech_token_len}" -v max_length="${max_speech_token_len}" \
                '{ if ($2 > min_length && $2 < max_length) print($0); }' \
                >"${data_feats}/${dset}/utt2num_frames.${speech_token_lang}"
        utils/filter_scp.pl \
            "${data_feats}/${dset}/utt2num_frames.${speech_token_lang}" \
            "${data_feats}/${dset}/utt2spk" > "${data_feats}/${dset}/utt2spk.tmp"
        mv ${data_feats}/${dset}/utt2spk.tmp ${data_feats}/${dset}/utt2spk

        # Remove empty text
        <"${data_feats}/org/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}" \
            awk ' { if ( NF != 1 ) print($0); } ' > "${data_feats}/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}"
        utils/filter_scp.pl \
            "${data_feats}/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}" \
            ${data_feats}/${dset}/utt2spk \
            > ${data_feats}/${dset}/utt2spk.tmp
        mv ${data_feats}/${dset}/utt2spk.tmp ${data_feats}/${dset}/utt2spk
        <"${data_feats}/org/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}" \
            awk ' { if ( NF != 1 ) print($0); } ' > "${data_feats}/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}"
        utils/filter_scp.pl \
            "${data_feats}/${dset}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_}" \
            ${data_feats}/${dset}/utt2spk \
            > ${data_feats}/${dset}/utt2spk.tmp
        mv ${data_feats}/${dset}/utt2spk.tmp ${data_feats}/${dset}/utt2spk

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh \
            --utt_extra_files "text.${speech_token_case}.${speech_token_lang} text.${src_tgt_text_case}.${src_tgt_text_lang//\//_} text.${src_tgt_text_case}.ctc" \
            "${data_feats}/${dset}"
    done

    # shellcheck disable=SC2002
    cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && ! [[ " ${skip_stages} " =~ [[:space:]]7[[:space:]] ]]; then

    if "${token_joint}"; then
        log "Merge src and target data if joint BPE"

        _tgt_bpe_train_text="${data_feats}/${train_set}/text.${speech_token_lang}_${src_tgt_text_lang//\//_}"
        cat $tgt_bpe_train_text > ${_tgt_bpe_train_text}
        [ -n "${src_bpe_train_text}" ] && cat ${src_bpe_train_text} >> ${_tgt_bpe_train_text}
        # Set the new text as the target text
        tgt_bpe_train_text=${_tgt_bpe_train_text}
    fi

    # First generate target text
    if [ "${tgt_token_type}" = bpe ]; then
        log "Stage 7a: Generate token_list from ${tgt_bpe_train_text} using BPE for target"

        mkdir -p "${tgt_bpedir}"
        # shellcheck disable=SC2002
        cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${tgt_bpedir}"/train.txt

        user_defined_symbols=
        if [ -n "${tgt_bpe_nlsyms}" ]; then
            if test -f "${tgt_bpe_nlsyms}"; then
                tgt_bpe_nlsyms_list=$(awk '{print $1}' ${tgt_bpe_nlsyms} | paste -s -d, -)
                user_defined_symbols+="${tgt_bpe_nlsyms_list}"
            else
                user_defined_symbols+="${tgt_bpe_nlsyms}"
            fi
            user_defined_symbols+=","
        fi
        user_defined_symbols+=$(echo ${tgt_tasks} | tr "/" "," | sed -re 's/\w+/<&>/g')
        user_defined_symbols+=$(echo ","${src_tgt_text_lang} | tr "/" "," | sed -re 's/\w+/<&>/g')
        if [ -n "${not_avail}" ]; then
            user_defined_symbols+=",${not_avail}"
        fi
        _opts_spm="--user_defined_symbols=${user_defined_symbols}"

        spm_train \
            --input="${tgt_bpedir}"/train.txt \
            --vocab_size="${tgt_nbpe}" \
            --model_type="${tgt_bpemode}" \
            --model_prefix="${tgt_bpeprefix}" \
            --character_coverage=${tgt_bpe_char_cover} \
            --input_sentence_size="${tgt_bpe_input_sentence_size}" \
            ${_opts_spm}

        {
        echo "${blank}"
        echo "${oov}"
        # Remove <unk>, <s>, </s> from the vocabulary
        <"${tgt_bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${sos_eos}"
        } > "${tgt_token_list}"

    elif [ "${tgt_token_type}" = char ] || [ "${tgt_token_type}" = word ]; then
        log "Stage 7a: Generate character level token_list from ${tgt_bpe_train_text} for ${src_tgt_text_lang}"

        if [ "${nlsyms_txt}" != none ]; then
            for task_id in $(echo ${tgt_tasks} | tr "/" " "); do
                echo ${task_id} >> ${nlsyms_txt}
            done
        else
            nlsyms_txt="data/nlsyms.txt"
            touch ${nlsyms_txt}
            for task_id in $(echo ${tgt_tasks} | tr "/" " "); do
                echo "<${task_id}>" >> ${nlsyms_txt}
            done
        fi
        if [ -n "${not_avail}" ]; then
            echo ${not_avail} >> ${nlsyms_txt}
        fi

        _opts="--non_linguistic_symbols ${nlsyms_txt}"

        # shellcheck disable=SC2002
        cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ST and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text  \
            --token_type "${tgt_token_type}" \
            --input "${data_feats}/token_train.txt" --output "${tgt_token_list}" ${_opts} \
            --field 1- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"

    else
        log "Error: not supported --token_type '${tgt_token_type}'"
        exit 2
    fi

    # Create word-list for word-LM training
    if ${use_word_lm} && [ "${tgt_token_type}" != word ]; then
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

    # Then generate for source data
    if "${token_joint}"; then
        log "Stage 7b: Skip separate token construction for source when setting ${token_joint} as true"
    else
        if [ "${src_token_type}" = bpe ]; then
            log "Stage 7b: Generate token_list from ${src_bpe_train_text} using BPE for source data"

            mkdir -p "${src_bpedir}"
            # shellcheck disable=SC2002
            cat ${src_bpe_train_text} | cut -f 2- -d" "  > "${src_bpedir}"/train.txt

            if [ -n "${src_bpe_nlsyms}" ]; then
                if test -f "${src_bpe_nlsyms}"; then
                    src_bpe_nlsyms_list=$(awk '{print $1}' ${src_bpe_nlsyms} | paste -s -d, -)
                    _opts_spm="--user_defined_symbols=${src_bpe_nlsyms_list}"
                else
                    _opts_spm="--user_defined_symbols=${src_bpe_nlsyms}"
                fi
            else
                _opts_spm=""
            fi

            spm_train \
                --input="${src_bpedir}"/train.txt \
                --vocab_size="${src_nbpe}" \
                --model_type="${src_bpemode}" \
                --model_prefix="${src_bpeprefix}" \
                --character_coverage=${src_bpe_char_cover} \
                --input_sentence_size="${src_bpe_input_sentence_size}" \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${src_bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${src_token_list}"

        elif [ "${src_token_type}" = char ] || [ "${src_token_type}" = word ]; then
            log "Stage 7b: Generate character level token_list from ${src_bpe_train_text} for input data"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # shellcheck disable=SC2002
            cat ${src_bpe_train_text} | tr '\t' ' ' | cut -f 2- -d" "  > "${data_feats}"/token_train_${speech_token_lang}.txt

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for ST and also used as ignore-index in the other task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${src_token_type}" \
                --input "${data_feats}/token_train_${speech_token_lang}.txt" --output "${src_token_list}" ${_opts} \
                --field 1- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"

        else
            log "Error: not supported --token_type '${src_token_type}'"
            exit 2
        fi
    fi
fi


# ========================== Data preparation is done here. ==========================


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ] && ! [[ " ${skip_stages} " =~ [[:space:]]8[[:space:]] ]]; then
    log "Stage 8: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

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
            --bpemodel "${tgt_bpemodel}" \
            --token_type "${lm_token_type}"\
            --token_list "${lm_token_list}" \
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
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

    <"${lm_stats_dir}/valid/text_shape" \
        awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
        >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    log "Stage 9: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

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
                --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                --num_splits "${num_splits_lm}" \
                --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.txt,text,text "
        _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
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
            --bpemodel "${tgt_bpemodel}" \
            --token_type "${lm_token_type}"\
            --token_list "${lm_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
            --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
            --fold_length "${lm_fold_length}" \
            --resume true \
            --output_dir "${lm_exp}" \
            ${_opts} ${lm_args}

fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Calc perplexity: ${lm_test_text}"
    _opts=
    # TODO(kamo): Parallelize?
    log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
        ${python} -m espnet2.bin.lm_calc_perplexity \
            --ngpu "${ngpu}" \
            --data_path_and_name_and_type "${lm_test_text},text,text" \
            --train_config "${lm_exp}"/config.yaml \
            --model_file "${lm_exp}/${inference_lm}" \
            --output_dir "${lm_exp}/perplexity_test" \
            ${_opts}
    log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ] && ! [[ " ${skip_stages} " =~ [[:space:]]11[[:space:]] ]]; then
    log "Stage 11: Ngram Training: train_set=${data_feats}/lm_train.txt"
    mkdir -p ${ngram_exp}
    cut -f 2- -d " " ${data_feats}/lm_train.txt | lmplz -S "20%" --discount_fallback -o ${ngram_num} - >${ngram_exp}/${ngram_num}gram.arpa
    build_binary -s ${ngram_exp}/${ngram_num}gram.arpa ${ngram_exp}/${ngram_num}gram.bin
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ] && ! [[ " ${skip_stages} " =~ [[:space:]]12[[:space:]] ]]; then
    _st_train_dir="${data_feats}/${train_set}"
    _st_valid_dir="${data_feats}/${valid_set}"
    log "Stage 12: ST collect stats: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

    _opts=
    if [ -n "${st_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.mt_train --print_config --optim adam
        _opts+="--config ${st_config} "
    fi

    # 1. Split the key file
    _logdir="${st_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    _scp=text.${speech_token_case}.${speech_token_lang}

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_st_train_dir}/${_scp} wc -l)" "$(<${_st_valid_dir}/${_scp} wc -l)")

    key_file="${_st_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_st_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${st_stats_dir}/run.sh'. You can resume the process from stage 12 using this script"
    mkdir -p "${st_stats_dir}"; echo "${run_args} --stage 9 \"\$@\"; exit \$?" > "${st_stats_dir}/run.sh"; chmod +x "${st_stats_dir}/run.sh"

    # 3. Submit jobs
    log "ST collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    # shellcheck disable=SC2068
    for i in ${!utt_extra_files[@]}; do
        if [ "${utt_extra_files[$i]}" = "utt2category" ]; then
            continue
        elif [ "${utt_extra_files_names[$i]}" = "text_ctc" ]; then
            _opts+="--ctc_bpemodel ${tgt_bpemodel} "
            _opts+="--ctc_token_type ${tgt_token_type} "
            _opts+="--ctc_token_list ${tgt_token_list} "
        fi
        _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/${utt_extra_files[$i]},${utt_extra_files_names[$i]},text "
        _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/${utt_extra_files[$i]},${utt_extra_files_names[$i]},text "
    done

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.mt_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${tgt_bpemodel}" \
            --src_bpemodel "${src_bpemodel}" \
            --token_type "${tgt_token_type}" \
            --src_token_type "${src_token_type}" \
            --token_list "${tgt_token_list}" \
            --src_token_list "${src_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${st_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${st_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    # shellcheck disable=SC2068
    for i in ${!utt_extra_files[@]}; do
        if [ ${utt_extra_files[i]} = "utt2category" ]; then
            continue
        elif [[ ${utt_extra_files[$i]} =~ ^"src" ]]; then
            _token_list="${src_token_list}"
            _token_type="${src_token_type}"
        else
            _token_list="${tgt_token_list}"
            _token_type="${tgt_token_type}"
        fi

        <"${st_stats_dir}/train/${utt_extra_files_names[$i]}_shape" \
            awk -v N="$(<${_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/train/${utt_extra_files_names[$i]}_shape.${_token_type}"

        <"${st_stats_dir}/valid/${utt_extra_files_names[$i]}_shape" \
            awk -v N="$(<${_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/valid/${utt_extra_files_names[$i]}_shape.${_token_type}"
    done
fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ] && ! [[ " ${skip_stages} " =~ [[:space:]]13[[:space:]] ]]; then
    _st_train_dir="${data_feats}/${train_set}"
    _st_valid_dir="${data_feats}/${valid_set}"
    log "Stage 13: ST Training: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

    _opts=
    if [ -n "${st_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.mt_train --print_config --optim adam
        _opts+="--config ${st_config} "
    fi

    if [ "${num_splits_st}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${st_stats_dir}/splits${num_splits_st}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"

            _scps=
            _scps+="${_st_train_dir}/${_scp} "
            _scps+="${_st_train_dir}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_} "
            _scps+="${_st_train_dir}/text.${speech_token_case}.${speech_token_lang} "
            _scps+="${st_stats_dir}/train/text_shape.${tgt_token_type} "
            _scps+="${st_stats_dir}/train/src_text_shape.${src_token_type} "
            if echo "${utt_extra_files_names[@]}" | grep -qw "text_ctc"; then
                _scps+="${_st_train_dir}/text.${src_tgt_text_case}.ctc "
                _scps+="${_st_train_dir}/text_ctc_shape.${tgt_token_type} "
            fi

            ${python} -m espnet2.bin.split_scps \
                --scps ${_scps} \
                --num_splits "${num_splits_st}" \
                --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_tgt_text_case}.${src_tgt_text_lang//\//_},text,text "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${speech_token_case}.${speech_token_lang},src_text,text "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${tgt_token_type} "
        _opts+="--train_shape_file ${_split_dir}/src_text_shape.${src_token_type} "
        if echo "${utt_extra_files_names[@]}" | grep -qw "text_ctc"; then
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_tgt_text_case}.ctc,text,text "
            _opts+="--train_shape_file ${_split_dir}/text_ctc_shape.${tgt_token_type} "
        fi
        _opts+="--multiple_iterator true "
    else
        # shellcheck disable=SC2068
        for i in ${!utt_extra_files[@]}; do
            if [ ${utt_extra_files[i]} = "utt2category" ]; then
                continue
            elif [[ "${utt_extra_files_names[i]}" =~ ^"src" ]]; then
                _token_type="${src_token_type}"
            else
                _token_type="${tgt_token_type}"
                if [ "${utt_extra_files_names[$i]}" = "text_ctc" ]; then
                    _opts+="--ctc_bpemodel ${tgt_bpemodel} "
                    _opts+="--ctc_token_type ${tgt_token_type} "
                    _opts+="--ctc_token_list ${tgt_token_list} "
                fi
            fi
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/${utt_extra_files[$i]},${utt_extra_files_names[$i]},text "
            _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/${utt_extra_files[$i]},${utt_extra_files_names[$i]},text "
            _opts+="--train_shape_file ${st_stats_dir}/train/${utt_extra_files_names[$i]}_shape.${_token_type} "
            _opts+="--valid_shape_file ${st_stats_dir}/valid/${utt_extra_files_names[$i]}_shape.${_token_type} "
            _opts+="--fold_length ${st_text_fold_length} "
        done
    fi

    log "Generate '${st_exp}/run.sh'. You can resume the process from stage 10 using this script"
    mkdir -p "${st_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${st_exp}/run.sh"; chmod +x "${st_exp}/run.sh"

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
    log "ST training started... log: '${st_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${st_exp})"
    else
        jobname="${st_exp}/train.log"
    fi

    # TODO(jiatong): fix bpe
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${st_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${st_exp}"/.dimt_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.mt_train \
            --use_preprocessor true \
            --bpemodel "${tgt_bpemodel}" \
            --token_type "${tgt_token_type}" \
            --token_list "${tgt_token_list}" \
            --src_bpemodel "${src_bpemodel}" \
            --src_token_type "${src_token_type}" \
            --src_token_list "${src_token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --resume true \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --output_dir "${st_exp}" \
            ${_opts} ${st_args}

fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    st_exp="${expdir}/${download_model}"
    mkdir -p "${st_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${st_exp}/config.txt"

    # Get the path of each file
    _st_model_file=$(<"${st_exp}/config.txt" sed -e "s/.*'st_model_file': '\([^']*\)'.*$/\1/")
    _st_train_config=$(<"${st_exp}/config.txt" sed -e "s/.*'st_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_st_model_file}" "${st_exp}"
    ln -sf "${_st_train_config}" "${st_exp}"
    inference_st_model=$(basename "${_st_model_file}")

    if [ "$(<${st_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${st_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${st_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi


if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ] && ! [[ " ${skip_stages} " =~ [[:space:]]14[[:space:]] ]]; then
    log "Stage 14: Decoding: training_dir=${st_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _lm_opts=
    if "${use_lm}"; then
        if "${use_word_lm}"; then
            _lm_opts+="--word_lm_train_config ${lm_exp}/config.yaml "
            _lm_opts+="--word_lm_file ${lm_exp}/${inference_lm} "
        else
            _lm_opts+="--lm_train_config ${lm_exp}/config.yaml "
            _lm_opts+="--lm_file ${lm_exp}/${inference_lm} "
        fi
    fi
    if "${use_ngram}"; then
        _lm_opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
    fi

    tgt_tasks_lst=(${tgt_tasks//\// })
    for i in ${!tgt_tasks_lst[@]}; do
        task_id=${tgt_tasks_lst[i]}

        _inference_config=
        _inference_tag=
        if [ "${task_id}" = "asr" ]; then
            if [ -n "${inference_asr_config}" ]; then
                _inference_config="${inference_asr_config} "
            fi
            _inference_tag="${inference_asr_tag}"
            _scp=text.${speech_token_case}.${speech_token_lang}
        elif [ "${task_id}" = "mt" ]; then
            if [ -n "${inference_mt_config}" ]; then
                _inference_config="${inference_mt_config} "
            fi
            _inference_tag="${inference_mt_tag}"
            _scp=text.${src_tgt_text_case}.${src_tgt_lang_lst[0]}
        elif [ "${task_id}" = "st" ]; then
            if [ -n "${inference_st_config}" ]; then
                _inference_config="${inference_st_config} "
            fi
            _inference_tag="${inference_st_tag}"
            _scp=text.${speech_token_case}.${speech_token_lang}
        else
            echo "Unsupported task_id: ${task_id}" && exit 1;
        fi
        _opts=${_lm_opts}
        if [ -n "${_inference_config}" ]; then
            _opts+="--config ${_inference_config} "
        fi
        _opts+="--task_id \"<${task_id}>\""

        # 2. Generate run.sh
        log "Generate '${st_exp}/${task_id}_${_inference_tag}/run.sh'. You can resume the process from stage 14 using this script"
        mkdir -p "${st_exp}/${task_id}_${_inference_tag}"; echo "${run_args} --stage 14 \"\$@\"; exit \$?" > "${st_exp}/${task_id}_${_inference_tag}/run.sh"; chmod +x "${st_exp}/${task_id}_${_inference_tag}/run.sh"

        for dset in ${test_sets}; do  # original valid_set has been merged into test_sets if eval_valid_set
            _data="${data_feats}/${dset}"
            _dir="${st_exp}/${task_id}_${_inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            st_inference_tool="espnet2.bin.mt_inference"

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/${task_id}_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/${task_id}_inference.JOB.log \
                ${python} -m ${st_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},src_text,text" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --mt_train_config "${st_exp}"/config.yaml \
                    --mt_model_file "${st_exp}"/"${inference_st_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/${task_id}_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    done
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ] && ! [[ " ${skip_stages} " =~ [[:space:]]15[[:space:]] ]]; then
    log "Stage 15a: ASR Scoring"

    _src_lang=${src_tgt_lang_lst[0]}
    _tgt_lang=${src_tgt_lang_lst[1]}

    task_id="asr"   # ASR task scoring only
    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${st_exp}/${task_id}_${inference_asr_tag}/${dset}"

        for _tok_type in "char" "word" "bpe"; do
            [ "${_tok_type}" = bpe ] && [ ! -f "${tgt_bpemodel}" ] && continue

            _opts="--token_type ${_tok_type} "
            if [ "${_tok_type}" = "char" ] || [ "${_tok_type}" = "word" ]; then
                _type="${_tok_type:0:1}er"
                _opts+="--non_linguistic_symbols ${nlsyms_txt} "
                _opts+="--remove_non_linguistic_symbols true "

            elif [ "${_tok_type}" = "bpe" ]; then
                _type="ter"
                _opts+="--bpemodel ${tgt_bpemodel} "

            else
                log "Error: unsupported token type ${_tok_type}"
            fi

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            # Tokenize text to ${_tok_type} level
            paste \
                <(<"${_data}/text.${src_tgt_text_case}.${_src_lang}" \
                    ${python} -m espnet2.bin.tokenize_text  \
                        -f 2- --input - --output - \
                        --token_type ${_tok_type} \
                        --cleaner "${cleaner}" \
                        ${_opts} \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/ref.trn"

            # NOTE(kamo): Don't use cleaner for hyp
            paste \
                <(<"${_dir}/text"  \
                    ${python} -m espnet2.bin.tokenize_text  \
                        -f 2- --input - --output - \
                        --token_type ${_tok_type} \
                        ${_opts} \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/hyp.trn"

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done
    done

    [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${st_exp}"

    # Show results in Markdown syntax
    scripts/utils/show_asr_result.sh "${st_exp}" > "${st_exp}"/RESULTS_asr.md
    cat "${st_exp}"/RESULTS_asr.md

    log "Stage 15b: MT / ST Scoring"

    for task_id in ${tgt_tasks//\// }; do
        if [ ${task_id} = "asr" ]; then
            continue
        elif [ ${task_id} = "st" ]; then
            _inference_tag="${inference_st_tag}"
        elif [ ${task_id} = "mt" ]; then
            _inference_tag="${inference_mt_tag}"
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${st_exp}/${task_id}_${_inference_tag}/${dset}"

            _scoredir="${_dir}/score_bleu"
            mkdir -p "${_scoredir}"

            paste \
                <(<"${_data}/text.${src_tgt_text_case}.${_tgt_lang}" \
                    ${python} -m espnet2.bin.tokenize_text \
                        -f 2- --input - --output - \
                        --token_type word \
                        --non_linguistic_symbols "${nlsyms_txt}" \
                        --remove_non_linguistic_symbols true \
                        --cleaner "${cleaner}" \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/ref.trn.org"

            # NOTE(kamo): Don't use cleaner for hyp
            paste \
                <(<"${_dir}/text" \
                        ${python} -m espnet2.bin.tokenize_text  \
                            -f 2- --input - --output - \
                            --token_type word \
                            --non_linguistic_symbols "${nlsyms_txt}" \
                            --remove_non_linguistic_symbols true \
                            ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/hyp.trn.org"

            # remove utterance id
            perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
            perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

            # detokenizer
            detokenizer.perl -l ${_tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
            detokenizer.perl -l ${_tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

            # rotate result files
            if [ ${src_tgt_text_case} = "tc" ]; then
                pyscripts/utils/rotate_logfile.py ${_scoredir}/result.tc.txt
            fi
            pyscripts/utils/rotate_logfile.py ${_scoredir}/result.lc.txt

            if [ ${src_tgt_text_case} = "tc" ]; then
                echo "Case sensitive BLEU result (single-reference)" > ${_scoredir}/result.tc.txt
                sacrebleu "${_scoredir}/ref.trn.detok" \
                            -i "${_scoredir}/hyp.trn.detok" \
                            -m bleu chrf ter \
                            >> ${_scoredir}/result.tc.txt

                log "Write a case-sensitive BLEU (single-reference) result in ${_scoredir}/result.tc.txt"
            fi

            # detokenize & remove punctuation except apostrophe
            scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
            scripts/utils/remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"
            echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
            sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                        -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                        -m bleu chrf ter \
                        >> ${_scoredir}/result.lc.txt
            log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"

            # process multi-references cases
            multi_references=$(ls "${_data}/text.${src_tgt_text_case}.${_tgt_lang}".* || echo "")
            if [ "${multi_references}" != "" ]; then
                case_sensitive_refs=""
                case_insensitive_refs=""
                for multi_reference in ${multi_references}; do
                    ref_idx="${multi_reference##*.}"
                    paste \
                        <(<${multi_reference} \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn.org.${ref_idx}"

                    # remove utterance id
                    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org.${ref_idx}" > "${_scoredir}/ref.trn.${ref_idx}"
                    detokenizer.perl -l ${_tgt_lang} -q < "${_scoredir}/ref.trn.${ref_idx}" > "${_scoredir}/ref.trn.detok.${ref_idx}"
                    scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok.${ref_idx}" > "${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                    case_sensitive_refs="${case_sensitive_refs} ${_scoredir}/ref.trn.detok.${ref_idx}"
                    case_insensitive_refs="${case_insensitive_refs} ${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                done

                if [ ${src_tgt_text_case} = "tc" ]; then
                    echo "Case sensitive BLEU result (multi-references)" >> ${_scoredir}/result.tc.txt
                    sacrebleu ${case_sensitive_refs} \
                        -i ${_scoredir}/hyp.trn.detok.lc.rm -m bleu chrf ter \
                        >> ${_scoredir}/result.tc.txt
                    log "Write a case-sensitve BLEU (multi-reference) result in ${_scoredir}/result.tc.txt"
                fi

                echo "Case insensitive BLEU result (multi-references)" >> ${_scoredir}/result.lc.txt
                sacrebleu -lc ${case_insensitive_refs} \
                    -i ${_scoredir}/hyp.trn.detok.lc.rm -m bleu chrf ter \
                    >> ${_scoredir}/result.lc.txt
                log "Write a case-insensitve BLEU (multi-reference) result in ${_scoredir}/result.lc.txt"
            fi
        done

        # Show results in Markdown syntax
        scripts/utils/show_translation_result.sh --case ${src_tgt_text_case} "${st_exp}" > "${st_exp}"/RESULTS_${task_id}.md
        cat "${st_exp}"/RESULTS_${task_id}.md
    done
fi


packed_model="${st_exp}/${st_exp##*/}_${inference_st_model%.*}.zip"
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ] && ! [[ " ${skip_stages} " =~ [[:space:]]16[[:space:]] ]]; then
    log "Stage 16: Pack model: ${packed_model}"

    _opts=
    if "${use_lm}"; then
        _opts+="--lm_train_config ${lm_exp}/config.yaml "
        _opts+="--lm_file ${lm_exp}/${inference_lm} "
        _opts+="--option ${lm_exp}/perplexity_test/ppl "
        _opts+="--option ${lm_exp}/images "
    fi
    if [ "${tgt_token_type}" = bpe ]; then
        _opts+="--option ${tgt_bpemodel} "
        _opts+="--option ${src_bpemodel} "
    fi
    if [ "${nlsyms_txt}" != none ]; then
        _opts+="--option ${nlsyms_txt} "
    fi
    _km_dir="exp/kmeans/$(echo ${kmeans_feature} | tr '/' '_')_${nclusters}clusters"
    _opts+="--option ${_km_dir}/km_${nclusters}.mdl "
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack asr \
        --asr_train_config "${st_exp}"/config.yaml \
        --asr_model_file "${st_exp}"/"${inference_st_model}" \
        ${_opts} \
        --option "${st_exp}"/RESULTS.md \
        --option "${st_exp}"/images \
        --outpath "${packed_model}"
fi


if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ] && ! [[ " ${skip_stages} " =~ [[:space:]]17[[:space:]] ]]; then
    log "Stage 17: Upload model to Zenodo: ${packed_model}"
    log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

    # To upload your model, you need to do:
    #   1. Sign up to Zenodo: https://zenodo.org/
    #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
    #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

    if command -v git &> /dev/null; then
        _creator_name="$(git config user.name)"
        _checkout="
git checkout $(git show -s --format=%H)"

    else
        _creator_name="$(whoami)"
        _checkout=""
    fi
    # /some/where/espnet/egs2/foo/st2/ -> foo/st2
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/st2 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # Generate description file
    cat << EOF > "${st_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${st_exp}"/RESULTS.md)</code></pre></li>
<li><strong>ST config</strong><pre><code>$(cat "${st_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

    # NOTE(kamo): The model file is uploaded here, but not published yet.
    #   Please confirm your record at Zenodo and publish it by yourself.

    # shellcheck disable=SC2086
    espnet_model_zoo_upload \
        --file "${packed_model}" \
        --title "ESPnet2 pretrained model, ${_model_name}, lang=${lang}" \
        --description_file "${st_exp}"/description \
        --creator_name "${_creator_name}" \
        --license "CC-BY-4.0" \
        --use_sandbox false \
        --publish false
fi


if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ] && ! [[ " ${skip_stages} " =~ [[:space:]]18[[:space:]] ]]; then
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1
    log "Stage 18: Upload model to HuggingFace: ${hf_repo}"

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
    # /some/where/espnet/egs2/foo/st2/ -> foo/st2
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/st2 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=automatic-speech-recognition
    # shellcheck disable=SC2034
    espnet_task=ASR
    # shellcheck disable=SC2034
    task_exp=${st_exp}
    # shellcheck disable=SC2034
    lang=${tgt_lang}
    eval "echo \"$(cat scripts/utils/TEMPLATE_HF_Readme.md)\"" > "${dir_repo}"/README.md

    this_folder=${PWD}
    cd ${dir_repo}
    if [ -n "$(git status --porcelain)" ]; then
        git lfs track *.mdl
        git add .
        git commit -m "Update model"
    fi
    git push
    cd ${this_folder}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
