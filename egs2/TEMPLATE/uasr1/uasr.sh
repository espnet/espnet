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
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_collect_stats=true # Whether to perform gpu collect stats
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or extracted).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.
vad_home=            # The directory for vad
                     # TODO(jiatong): add more options and variants of vad for choices
silence_trim=true    # Whether to apply vad information in audio trimming
precompute_batchsize=1       # Batchsize for feature pre-computation
write_collected_feats=false  # Whether to write collected feats for faster training (need more extra spaces)

# Tokenization related
token_type=phn      # Tokenization type (phn).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
reduce_vocab=true   # Reduce vocabulary size by removing suffix digits in tokens
oov="<unk>"         # Out of vocabulary symbol.
blank="<eps>"       # CTC blank symbol/ WFST espilon symbol
sos="<s>"           # sos symbol
eos="</s>"          # eos symbol
pad="<pad>"         # padding symbol
sos_eos="<sos/eos>" # sos and eos symbol
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma or a file containing 1 symbol per line, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE
postprocess_word_boundary="   "  # word boundary for post process
postprocess_sil_token="<SIL>"  # silence injection tokens in post process
postprocess_sil_prob=0.5         # silence injection probability in post process

# Ngram language model related
use_ngram=true      # Whether to use n-gram modeling (Current version must set to true)
ngram_exp=          # Specify the diretory path for ngram LM experiments
ngram_num=4         # Specify the n in ngram LM
kenlm_path=         # Pre-trained ngram LM

# Language model related
use_lm=true       # Use language model for uasr decoding.
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

# UASR V1 feature clustering related
use_feature_clustering=false        # Do PCA and k-means on input feature
feature_clustering_tool="faiss"     # Tool for feature clustering (faiss or cuml)
feature_pca_dim=512                 # Dimension of PCAed feature vector
feature_num_clusters=128            # Number of feature clusters

# uasr model related
uasr_tag=       # Suffix to the result dir for uasr model training.
uasr_exp=       # Specify the directory path for uasr experiment.
               # If this option is specified, uasr_tag is ignored.
uasr_stats_dir= # Specify the directory path for uasr statistics.
uasr_config=    # Config for uasr model training.
uasr_args=      # Arguments for uasr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in uasr config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
num_splits_uasr=1           # Number of splitting for lm corpus.


# k2-decoding related
use_k2=false  # Whether to use k2-based decoder.
k2_lexicon=   # Specify a lexicon for k2-based decoding.
k2_lang_dir=  # Specify a directory to store lexicon and graphs for k2-based decoding.
k2_graph_dir= # Specify the HLG graph directory.
k2_config=conf/decode_uasr_k2.yaml # Detailed configurations for k2-based decoding.

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin  # Ngram language model path for decoding.
inference_uasr_model=valid.weighted_lm_ppl.best.pth # uasr model path for decoding.
                                                    # e.g.
                                                    # inference_uasr_model=train.loss.best.pth
                                                    # inference_uasr_model=3epoch.pth
download_model= # Download a model from Model Zoo and use it for decoding.
fairseq_checkpoint= # Decode with a fairseq pre-trained checkpoint.

# Scoring related
remove_silence=true # Remove silence token in hyp

# Upload model related
hf_repo=        # Specify a Huggingface directory.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
uasr_speech_fold_length=800 # fold_length for speech data during uasr training.
uasr_text_fold_length=150   # fold_length for text data during uasr training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --skip_upload_hf # Skip uploading to hugging face stages. (default="${skip_upload_hf}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_collect_stats  # Whether to perform gpu collect stats (default="${gpu_collect_stats}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --vad_home         # The directory for vad processing (default="${vad_home}").
    --silence_trim     # Whether to use vad to trim silence (default="${silence_trim}").
    --precompute_batchsize    # Batchsize for feature pre-computation (default="${precompute_batchsize}").
    --write_collected_feats   # Whether to write collected feats for faster training (default="${write_collected_feats}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma or a file containing 1 symbol per line . (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").
    --postprocess_word_boundary # word boundary for post process (default="${postprocess_word_boundary}").
    --postprocess_sil_token     # silence injection tokens in post process (default="${postprocess_sil_token}").
    --postprocess_sil_prob      # silence injection probability in post process (default="${postprocess_sil_prob}").

    # Ngram language model related
    --use_ngram       # Whether to use n-gram modeling (Current version must set to true)
    --ngram_exp       # Specify the diretory path for ngram LM experiments
    --ngram_num       # Specify the n in ngram LM
    --kenlm_path      # Specify the pre-trained LM path

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
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").

    # UASR V1 feature clustering related
    --use_feature_clustering # Whether conduct cluster for features (default="${use_feature_clustering}").
    --feature_clustering_tool # Tool to do feature clustering (default="${feature_clustering_tool}")
    --feature_pca_dim        # PCA dimension of features (default="${feature_pca_dim}").
    --feature_num_clusters   # Number of clusters for feature clustering pooling (default="${feature_num_clusters}").

    # uasr model related
    --uasr_tag          # Suffix to the result dir for uasr model training (default="${uasr_tag}").
    --uasr_exp          # Specify the directory path for uasr experiment.
                       # If this option is specified, uasr_tag is ignored (default="${uasr_exp}").
    --uasr_stats_dir    # Specify the directory path for uasr statistics (default="${uasr_stats_dir}").
    --uasr_config       # Config for uasr model training (default="${uasr_config}").
    --uasr_args         # Arguments for uasr model training (default="${uasr_args}").
                       # e.g., --uasr_args "--max_epoch 10"
                       # Note that it will overwrite args in uasr config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --num_splits_uasr   # Number of splitting for lm corpus  (default="${num_splits_uasr}").

    # k2-decoding related
    --use_k2            # Whether to use k2-based decoder (default="${use_k2}").
    --k2_lexicon        # Specify a lexicon for k2-based decoding (default="${k2_lexicon}").
    --k2_lang_dir       # Specify a directory to store lexicon and graphs for k2-based decoding (default="${k2_lang_dir}").
    --k2_graph_dir      # Specify the HLG graph directory (default="${k2_graph_dir}").
    --k2_config         # Detailed configurations for k2-based decoding (default="${k2_config}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_ngram     # Ngram anguage model path for decoding (default="${inference_lm}").
    --inference_uasr_model # uasr model path for decoding (default="${inference_uasr_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    --fairseq_checkpoint  # Decode with a fairseq pre-trained checkpoint (default="${fairseq_checkpoint}").

    # Model uploading related
    --hf_repo             # Specify a Huggingface directory (default="${hf_repo}")

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
    --uasr_speech_fold_length # fold_length for speech data during uasr training (default="${uasr_speech_fold_length}").
    --uasr_text_fold_length   # fold_length for text data during uasr training (default="${uasr_text_fold_length}").
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


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 3; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Use the same text as uasr for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as uasr for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as uasr for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"


# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi

unpaired_text="${data_feats}/${train_set}"/unpaired_text
unpaired_text_scp="${data_feats}/${train_set}"/unpaired_text.scp
unpaired_text_and_scp="${unpaired_text}-${unpaired_text_scp}"
bpemodel=none
if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
    tokendir="${token_listdir}/bpe_${bpemode}${nbpe}"
    bpeprefix="${tokendir}"/bpe
    bpemodel="${bpeprefix}".model
    bpetoken_list="${tokendir}"/tokens.txt
elif [ "${token_type}" = char ]; then
    tokendir="${token_listdir}"/char
    token_list="${tokendir}"/tokens.txt
elif [ "${token_type}" = phn ]; then
    tokendir="${token_listdir}"/"phn_${g2p}"
    token_list="${tokendir}"/tokens.txt
elif [ "${token_type}" = word ]; then
    # NOTE: keep for future development.
    # shellcheck disable=SC2034
    wordtoken_list="${token_listdir}"/word/tokens.txt
    tokendir="${token_listdir}"/word
    token_list="${wordtoken_list}"
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi

# Check if use k2 decoding
if ${use_k2}; then
    [ -z "${k2_lang_dir}" ] && k2_lang_dir="${tokendir}/lang"
    mkdir -p "${k2_lang_dir}"

    [ -z "${k2_graph_dir}" ] && k2_graph_dir="${tokendir}/graph"
    mkdir -p "${k2_graph_dir}"
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
if [ -z "${uasr_tag}" ]; then
    if [ -n "${uasr_config}" ]; then
        uasr_tag="$(basename "${uasr_config}" .yaml)_${feats_type}"
    else
        uasr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        uasr_tag+="_${lang}_${token_type}"
    else
        uasr_tag+="_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        uasr_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${uasr_args}" ]; then
        uasr_tag+="$(echo "${uasr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        uasr_tag+="_sp"
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
if [ -z "${uasr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        uasr_stats_dir="${expdir}/uasr_stats_${feats_type}_${lang}_${token_type}"
    else
        uasr_stats_dir="${expdir}/uasr_stats_${feats_type}_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        uasr_stats_dir+="${nbpe}"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        uasr_stats_dir+="_sp"
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
if [ -z "${uasr_exp}" ]; then
    uasr_exp="${expdir}/uasr_${uasr_tag}"
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
    inference_tag+="_uasr_model_$(echo "${inference_uasr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

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
        log "Stage 3: VAD computing for all data"
        if "${silence_trim}"; then
            # FIXME(jiatong): we now support limited wav format (i.e., only wav or flac for now) to process it
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                # FIXME(jiatong): only fs=16000 is supported now
                scripts/audio/compute_vad.sh  --cmd "${train_cmd}" --nj "${nj}" "${VAD_HOME}" "data/${dset}"
                utils/fix_data_dir.sh data/"${dset}"
            done
        else
            log "Skip stage 3: VAD computing for data"
        fi
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 4: Format wav.scp: data/ -> ${data_feats}"

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
                if "${silence_trim}"; then
                    _opts+="--vad_based_trim "data/${dset}/vad.scp" "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 4: ${feats_type} extract: data/ -> ${data_feats}"
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


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(${python} -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(${python} -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(${python} -c "print(int(${max_wav_duration} * ${_fs}))")

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

                _min_length=$(${python} -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(${python} -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

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

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done

    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _logdir="${tokendir}/logdir"

        # shellcheck disable=SC2002
        cat ${lm_train_text} | awk ' { if( NF != 2 ) print $0; } ' > "${data_feats}/lm_train.txt"

        if [ "${token_type}" = bpe ]; then
            log "Stage 6: Generate token_list from ${bpe_train_text} using BPE"

            log "Error: not supported --token_type '${token_type}'"
            exit 2

        elif [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 6: Generate character/word level token_list from ${data_feats}/lm_train.txt"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # split text for parallel tokenization
            split_dir=${tokendir}/split${nj}
            split_text=""
            mkdir -p ${split_dir}
            for n in $(seq ${nj}); do
                mkdir -p ${split_dir}/${n}
                split_text="${split_text} ${split_dir}/${n}/text"
            done

            utils/split_scp.pl "${data_feats}/lm_train.txt" ${split_text}

            ${train_cmd} JOB=1:${nj} ${_logdir}/lm/tokenize_text.JOB.log \
            ${python} -m espnet2.bin.tokenize_text \
                --token_type ${token_type} ${_opts} \
                --cleaner "${cleaner}" \
                --input ${split_dir}/JOB/text \
                --output ${split_dir}/JOB/unpaired_text \
                --write_vocabulary false \
                --field "2-"

            ${python} pyscripts/text/combine_text_and_vocab.py \
                    --split_dir ${split_dir} \
                    --num_splits ${nj} \
                    --output_dir "${data_feats}/${train_set}" \
                    --text_file "unpaired_text" \
                    --vocab_file "tokens.txt" \
                    --add_symbol "${blank}" \
                    --add_symbol "${sos}" \
                    --add_symbol "${pad}" \
                    --add_symbol "${eos}" \
                    --add_symbol "${oov}" \
                    --add_symbol "${postprocess_sil_token}"
            cp "${data_feats}/${train_set}/tokens.txt" "${token_list}"

            # Note(Dongji): for dev text we keep utterance ids to compute PER
            log "Tokenizing ${lm_dev_text}"
            cut -d ' ' -f1 "${lm_dev_text}" > "${data_feats}/${valid_set}/utt_ids"
            ${python} -m espnet2.bin.tokenize_text \
                --token_type ${token_type} ${_opts} \
                --cleaner "${cleaner}" \
                --input ${lm_dev_text}\
                --output "${data_feats}/${valid_set}/unpaired_text.tmp" \
                --write_vocabulary false \
                --field "2-"

            # Note(Jiatong): though name as unpaired text, we here utilize the paired data for internal
            # evaluation (however, the results with unpaired text are not suppose to be used for model
            # selection)
            paste -d ' ' "${data_feats}/${valid_set}/utt_ids" "${data_feats}/${valid_set}/unpaired_text.tmp" \
            > "${data_feats}/${valid_set}/unpaired_text"

        elif [ "${token_type}" == phn ]; then
            log "Stage 6: Generate phone level token_list from ${lm_train_text}"

            # split text
            split_dir=${tokendir}/split${nj}
            split_text=""
            mkdir -p ${split_dir}
            for n in $(seq ${nj}); do
                mkdir -p ${split_dir}/${n}
                split_text="${split_text} ${split_dir}/${n}/text"
            done

            utils/split_scp.pl "${data_feats}/lm_train.txt" ${split_text}

            ${train_cmd} JOB=1:${nj} ${_logdir}/lm/tokenize_text.JOB.log \
            ${python} -m espnet2.bin.tokenize_text \
                --token_type phn \
                --input ${split_dir}/JOB/text \
                --output ${split_dir}/JOB/unpaired_text_nosil \
                --g2p "${g2p}" \
                --write_vocabulary false \
                --field "2-"

             ${train_cmd} JOB=1:${nj} ${_logdir}/lm/post_processing.JOB.log \
             ${python} pyscripts/text/post_processing.py \
               --word_boundary "${postprocess_word_boundary}" \
               --sil_prob ${postprocess_sil_prob} \
               --sil_token "'${postprocess_sil_token}'" \
               --input_text "${split_dir}/JOB/unpaired_text_nosil" \
               --output_text "${split_dir}/JOB/unpaired_text" \
               --reduce_vocab ${reduce_vocab}

            ${python} pyscripts/text/combine_text_and_vocab.py \
                --split_dir ${split_dir} \
                --num_splits ${nj} \
                --output_dir "${data_feats}/${train_set}" \
                --text_file "unpaired_text" \
                --vocab_file "tokens.txt" \
                --add_symbol "${blank}" \
                --add_symbol "${sos}" \
                --add_symbol "${pad}" \
                --add_symbol "${eos}" \
                --add_symbol "${oov}"

            cp "${data_feats}/${train_set}/tokens.txt" "${token_list}"

            # Note(Dongji): for dev text we keep utterance ids to compute PER
            log "Tokenizing ${lm_dev_text}"
            cut -d ' ' -f1 "${lm_dev_text}" > "${data_feats}/${valid_set}/utt_ids"
            ${python} -m espnet2.bin.tokenize_text \
                --token_type phn \
                --input ${lm_dev_text}\
                --output "${data_feats}/${valid_set}/unpaired_text.raw" \
                --g2p "${g2p}" \
                --write_vocabulary false \
                --field "2-"

             ${python} pyscripts/text/post_processing.py \
               --word_boundary "${postprocess_word_boundary}" \
               --sil_prob 0.0 \
               --input_text "${data_feats}/${valid_set}/unpaired_text.raw" \
               --output_text "${data_feats}/${valid_set}/unpaired_text.tmp" \
               --reduce_vocab true

            paste -d ' ' "${data_feats}/${valid_set}/utt_ids" "${data_feats}/${valid_set}/unpaired_text.tmp" \
            > "${data_feats}/${valid_set}/unpaired_text"

        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
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
    if "${use_lm}"; then
        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            log "Stage 7: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

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
            log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 7 using this script"
            mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

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
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
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


        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

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

            log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 8 using this script"
            mkdir -p "${lm_exp}"; echo "${run_args} --stage 8 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

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
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --output_dir "${lm_exp}" \
                    ${_opts} ${lm_args}

        fi


        if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
            log "Stage 9: Calc perplexity: ${lm_test_text}"
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

    else
        log "Stage 7-9: Skip lm-related stages: use_lm=${use_lm}"
    fi


    if "${use_ngram}"; then
        mkdir -p ${ngram_exp}
        [ -z ${kenlm_path} ] && kenlm_path="${ngram_exp}/${ngram_num}gram.bin"
    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        if "${use_ngram}"; then
            log "Stage 10: Ngram Training: train_set=${unpaired_text}"
            _logdir="${ngram_exp}"
            lmplz -o ${ngram_num} < ${unpaired_text} --discount_fallback --prune 0 0 0 3 >${ngram_exp}/${ngram_num}gram.arpa
            build_binary ${ngram_exp}/${ngram_num}gram.arpa "${kenlm_path}"

            if "${use_k2}"; then
                log "Stage 10: Building text lm: train_set=${lm_train_text}"
                cut -d ' ' -f2- "${lm_train_text}" > "${k2_lang_dir}/text"
                lmplz -o ${ngram_num} < "${k2_lang_dir}/text" --discount_fallback --prune 0 0 0 3 \
                    >${ngram_exp}/${ngram_num}gram.word.arpa
            fi
        else
            log "Stage 10: Skip ngram stages: use_ngram=${use_ngram}"
        fi
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        if "${use_k2}"; then
            if [ -z "${k2_lexicon}" ]; then
                log "Stage 11: Making lexicon: train_set=${lm_train_text}"

                scripts/k2/make_lexicon.sh \
                    --text "${lm_train_text}" \
                    --lang_dir "${k2_lang_dir}" \
                    --g2p "${g2p}" \
                    --oov "${oov}" \
                    --reduce_vocab ${reduce_vocab}
            else
                [ ! -f "${k2_lang_dir}/lexicon.txt" ] && cp "${k2_lexicon}" "${k2_lang_dir}/lexicon.txt"
            fi

            log "Stage 11: Preparing lang directory"
            ${python} pyscripts/k2/prepare_lang.py \
              --lang_dir "${k2_lang_dir}" \
              --token_list "${token_list}" \
              --sil_token "${postprocess_sil_token}"

            ${python} -m kaldilm \
                --read-symbol-table="${k2_lang_dir}/words.txt" \
                --disambig-symbol="#0" \
                --max-order="${ngram_num}" \
                "${ngram_exp}/${ngram_num}gram.word.arpa" > "${k2_graph_dir}/G_${ngram_num}_gram.fst.txt"

            ${python} pyscripts/k2/compile_hlg.py \
                --lang_dir "${k2_lang_dir}" \
                --graph_dir "${k2_graph_dir}" \
                --ngram_num "${ngram_num}"
        fi
    else
        log "Stage 11: Skip building k2 graph for UASR training"
    fi

    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Build text scp file for RandomTextReader class"
        ${python} pyscripts/text/make_text_scp.py \
          --input_text ${unpaired_text} \
          --output_scp ${unpaired_text_scp} \
          --num_digits 11
    fi

    # TODO(Dongji): fix the ad-hoc token_type
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        _uasr_train_dir="${data_feats}/${train_set}"
        _uasr_valid_dir="${data_feats}/${valid_set}"

        log "Stage 13: UASR collect stats: train_set=${_uasr_train_dir}, valid_set=${_uasr_valid_dir}"

        if ${gpu_collect_stats}; then
            _cmd="${cuda_cmd}"
            _ngpu=${ngpu}
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        _opts+="--batch_size ${precompute_batchsize} "
        if [ -n "${uasr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.uasr_train --print_config --optim adam
            _opts+="--config ${uasr_config} "
        fi

        _feats_type="$(<${_uasr_train_dir}/feats_type)"
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
            _input_size="$(<${_uasr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${uasr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_uasr_train_dir}/${_scp} wc -l)" "$(<${_uasr_valid_dir}/${_scp} wc -l)")

        key_file="${_uasr_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_uasr_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${uasr_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${uasr_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${uasr_stats_dir}/run.sh"; chmod +x "${uasr_stats_dir}/run.sh"

        # 3. Submit jobs
        log "UASR collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2046,SC2086
        ${cuda_cmd} --gpu "${_ngpu}"  JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.uasr_train \
                --collect_stats true \
                --use_preprocessor true \
                --write_collected_feats ${write_collected_feats} \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --ngpu "${_ngpu}" \
                --train_data_path_and_name_and_type "${_uasr_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${unpaired_text_and_scp},text,random_text" \
                --valid_data_path_and_name_and_type "${_uasr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_uasr_valid_dir}/text,text,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${uasr_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${uasr_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${uasr_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${uasr_stats_dir}/train/text_shape.${token_type}"

        <"${uasr_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${uasr_stats_dir}/valid/text_shape.${token_type}"

        cp ${uasr_stats_dir}/train/collect_feats/feats.scp "${data_feats}/${train_set}"/feats.scp
        cp ${uasr_stats_dir}/valid/collect_feats/feats.scp "${data_feats}/${valid_set}"/feats.scp
    fi

    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: UASR Feature Preprocess: preprocess_feature_dumpdir=${uasr_stats_dir}"
        _logdir="${uasr_stats_dir}/logdir"

        # TODO(jiatong): generalize the clustering process (we have so many now, v1 preprocessing, pseudo labeling, auto-encoder)

        if "${use_feature_clustering}"; then
            output_feats_dir="${uasr_stats_dir}/clustered/"

            log "Using ${feature_clustering_tool} for feature clustering"
            if [ "${feature_clustering_tool}" = "faiss" ]; then
                scripts/feats/feats_clustering.sh \
                    --cmd "${cuda_cmd}" \
                    --nj ${nj} \
                    --dim ${feature_pca_dim} \
                    --num_clusters ${feature_num_clusters} \
                    ${uasr_stats_dir} \
                    ${output_feats_dir}

            elif [ "${feature_clustering_tool}" = "cuml" ]; then
                scripts/feats/feats_clustering_cuml.sh \
                    --cmd "${cuda_cmd}" \
                    --nj ${nj} \
                    --dim ${feature_pca_dim} \
                    --num_clusters ${feature_num_clusters} \
                    ${uasr_stats_dir} \
                    ${output_feats_dir}

            else
                log "${feature_clustering_tool}" is not supported
            fi

            echo "${feature_pca_dim}" > ${uasr_stats_dir}/train/feats_dim
            echo "${feature_pca_dim}" > ${uasr_stats_dir}/valid/feats_dim

            log "Using clustered feature"
            # TODO(Jiatong): remove hard-code paths
            cp "${uasr_stats_dir}/clustered/precompute_pca${feature_pca_dim}_cls${feature_num_clusters}_mean_pooled/train/feats.scp" \
                "${data_feats}/${train_set}"/feats.scp
            cp "${uasr_stats_dir}/clustered/precompute_pca${feature_pca_dim}_cls${feature_num_clusters}_mean_pooled/valid/feats.scp" \
                "${data_feats}/${valid_set}"/feats.scp
        fi
        # TODO(Jiatong): add pseudo label clustering
    else
        log "Stage 14: Skip doing feature preprocess"
    fi

    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        _uasr_train_dir="${data_feats}/${train_set}"
        _uasr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 15: UASR Training: train_set=${_uasr_train_dir}, valid_set=${_uasr_valid_dir}"

        _opts=
        if [ -n "${uasr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.uasr_train --print_config --optim adam
            _opts+="--config ${uasr_config} "
        fi

        _feats_type="$(<${_uasr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            if "${write_collected_feats}"; then
                # Whether to use extracted features from collect_stats
                _scp=feats.scp

                # TODO(jiatong): update other types for more compact version
                _type=npy
                _fold_length="${uasr_speech_fold_length}"
                _input_size="$(<${uasr_stats_dir}/train/feats_dim)"
                _opts+="--input_size=${_input_size} "
            else
                _scp=wav.scp
                # "sound" supports "wav", "flac", etc.
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
                _fold_length="$((uasr_speech_fold_length * 100))"
                _opts+="--frontend_conf fs=${fs} "
            fi
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${uasr_speech_fold_length}"
            _input_size="$(<${_uasr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        _type="npy"

        if [ "${num_splits_uasr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${uasr_stats_dir}/splits${num_splits_uasr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_uasr_train_dir}/${_scp}" \
                      "${_uasr_train_dir}/text" \
                      "${uasr_stats_dir}/train/speech_shape" \
                      "${uasr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_uasr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            # FIXME(Jiatong): this option is not tested for UASR
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${unpaired_text_and_scp},text,random_text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_uasr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${unpaired_text_and_scp},text,random_text "
            _opts+="--train_shape_file ${uasr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${uasr_stats_dir}/train/text_shape.${token_type} "
        fi
        [ ! -z ${fairseq_checkpoint} ] && _opts+="--fairseq_checkpoint ${fairseq_checkpoint}"

        log "Generate '${uasr_exp}/run.sh'. You can resume the process from stage 15 using this script"
        mkdir -p "${uasr_exp}"; echo "${run_args} --stage 15 \"\$@\"; exit \$?" > "${uasr_exp}/run.sh"; chmod +x "${uasr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "uasr training started... log: '${uasr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${uasr_exp})"
        else
            jobname="${uasr_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${uasr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${uasr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.uasr_train \
                --use_preprocessor true \
                --write_collected_feats ${write_collected_feats} \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --valid_data_path_and_name_and_type "${_uasr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_uasr_valid_dir}/unpaired_text,text,text" \
                --valid_shape_file "${uasr_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${uasr_stats_dir}/valid/text_shape.${token_type}" \
                --resume true \
                --init_param ${pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${_fold_length}" \
                --fold_length "${uasr_text_fold_length}" \
                --output_dir "${uasr_exp}" \
                --kenlm_path "${kenlm_path}" \
                ${_opts} ${uasr_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    uasr_exp="${expdir}/${download_model}"
    mkdir -p "${uasr_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${uasr_exp}/config.txt"

    # Get the path of each file
    _uasr_model_file=$(<"${uasr_exp}/config.txt" sed -e "s/.*'uasr_model_file': '\([^']*\)'.*$/\1/")
    _uasr_train_config=$(<"${uasr_exp}/config.txt" sed -e "s/.*'uasr_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_uasr_model_file}" "${uasr_exp}"
    ln -sf "${_uasr_train_config}" "${uasr_exp}"
    inference_uasr_model=$(basename "${_uasr_model_file}")

    if [ "$(<${uasr_exp}/config.txt grep -c lm_file)" -gt 0 ]; then
        _lm_file=$(<"${uasr_exp}/config.txt" sed -e "s/.*'lm_file': '\([^']*\)'.*$/\1/")
        _lm_train_config=$(<"${uasr_exp}/config.txt" sed -e "s/.*'lm_train_config': '\([^']*\)'.*$/\1/")

        lm_exp="${expdir}/${download_model}/lm"
        mkdir -p "${lm_exp}"

        ln -sf "${_lm_file}" "${lm_exp}"
        ln -sf "${_lm_train_config}" "${lm_exp}"
        inference_lm=$(basename "${_lm_file}")
    fi

fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
        log "Stage 16: Extracting feature: test_sets=${test_sets}"

        _logdir="${uasr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _scp=wav.scp
            _type=sound

            ${cuda_cmd} --gpu "${ngpu}" "${_logdir}/extract_feature_${dset}.log" \
                ${python} -m espnet2.bin.uasr_extract_feature \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --uasr_train_config "${uasr_exp}/config.yaml" \
                    --uasr_model_file "${uasr_exp}/${inference_uasr_model}" \
                    --key_file "${_data}/${_scp}"\
                    --ngpu ${ngpu} \
                    --batch_size ${precompute_batchsize} \
                    --output_dir "${uasr_stats_dir}" \
                    --dset "${dset}"

            cp ${uasr_stats_dir}/${dset}/collect_feats/feats.scp "${data_feats}/${dset}"/feats.scp
        done

        if "${use_feature_clustering}"; then
            output_feats_dir="${uasr_stats_dir}/clustered/"

            log "Using ${feature_clustering_tool} for feature clustering"
            if [ "${feature_clustering_tool}" = "faiss" ]; then
                scripts/feats/feats_clustering.sh \
                    --cmd "${cuda_cmd}" \
                    --nj ${nj} \
                    --dim ${feature_pca_dim} \
                    --num_clusters ${feature_num_clusters} \
                    --valid_set "" \
                    --test_sets "${test_sets}" \
                    --skip_training true \
                    ${uasr_stats_dir} \
                    ${output_feats_dir}

            elif [ "${feature_clustering_tool}" = "cuml" ]; then
                scripts/feats/feats_clustering_cuml.sh \
                    --cmd "${cuda_cmd}" \
                    --nj ${nj} \
                    --dim ${feature_pca_dim} \
                    --num_clusters ${feature_num_clusters} \
                    --valid_set "" \
                    --test_sets "${test_sets}" \
                    --skip_training true \
                    ${uasr_stats_dir} \
                    ${output_feats_dir}

            else
                log "${feature_clustering_tool}" is not supported
            fi

            for dset in ${test_sets}; do
                cp "${uasr_stats_dir}/clustered/precompute_pca${feature_pca_dim}_cls${feature_num_clusters}_mean_pooled/${dset}/feats.scp" \
                    "${data_feats}/${dset}"/feats.scp
            done
        fi
    fi

    if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
        log "Stage 17: Decoding: training_dir=${uasr_exp}"

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
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        # 2. Generate run.sh
        log "Generate '${uasr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${uasr_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${uasr_exp}/${inference_tag}/run.sh"; chmod +x "${uasr_exp}/${inference_tag}/run.sh"

        uasr_inference_tool="espnet2.bin.uasr_inference"
        if "${use_k2}"; then
            uasr_inference_tool="espnet2.bin.uasr_inference_k2"
            use_ngram=false

            _opts+="--k2_config ${k2_config} "
            _opts+="--token_type word "
            _opts+="--decoding_graph ${k2_graph_dir}/HLG.pt "
            _opts+="--word_token_list ${k2_lang_dir}/words.txt "
        fi

        if "${use_ngram}"; then
             _opts+="--ngram_file ${ngram_exp}/${inference_ngram}"
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${uasr_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                if "${write_collected_feats}"; then
                    # TODO(jiatong): TO remove this part, should prevent from using pre-extracted features for this purpose
                    # Whether to use extracted features from collect_stats
                    _scp=feats.scp
                    # TODO(jiatong): update other types for more compact version
                    _type=npy
                    _fold_length="${uasr_speech_fold_length}"
                else
                    _scp=wav.scp
                    # "sound" supports "wav", "flac", etc.
                    if [[ "${audio_format}" == *ark* ]]; then
                        _type=kaldi_ark
                    else
                        _type=sound
                    fi
                    _fold_length="$((uasr_speech_fold_length * 100))"
                    _opts+="--frontend_conf fs=${fs} "
                fi
            else
                _scp=feats.scp
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

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/uasr_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/uasr_inference.JOB.log \
                ${python} -m ${uasr_inference_tool} \
                    --batch_size 1 \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --uasr_train_config "${uasr_exp}"/config.yaml \
                    --uasr_model_file "${uasr_exp}/${inference_uasr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/uasr_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                  for i in $(seq "${_nj}"); do
                      cat "${_logdir}/output.${i}/1best_recog/${f}"
                  done | sort -k1 >"${_dir}/${f}"
                fi
            done
        done
    fi


    if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
        log "Stage 18: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${uasr_exp}/${inference_tag}/${dset}"

            if [ "${token_type}" = phn ]; then
                _scoredir="${_dir}/score_per"
                mkdir -p "${_scoredir}"

                log "${_data}/text"

                # Tokenize text to phn level (the phn is separate as word)
                ${python} -m espnet2.bin.tokenize_text \
                    --token_type phn \
                    --input "${_data}/text" \
                    --output "${_data}/phonemiced_text.tmp" \
                    --g2p "${g2p}" \
                    --field "2-"

                ${python} pyscripts/text/post_processing.py \
                    --word_boundary "${postprocess_word_boundary}" \
                    --sil_prob 0.0 \
                    --input_text "${_data}/phonemiced_text.tmp" \
                    --output_text "${_data}/phonemiced_text" \
                    --reduce_vocab ${reduce_vocab}

                paste \
                    <(cat "${_data}/phonemiced_text") \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

                sclite \
                    ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write PER result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"

            else

                for _type in cer wer ter; do
                    [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                    _scoredir="${_dir}/score_${_type}"
                    mkdir -p "${_scoredir}"

                    if [ "${_type}" = wer ]; then
                        # Tokenize text to word level
                        paste \
                            <(<"${_data}/text" \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type word \
                                    --non_linguistic_symbols "${nlsyms_txt}" \
                                    --remove_non_linguistic_symbols true \
                                    --cleaner "${cleaner}" \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/ref.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text"  \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type word \
                                    --non_linguistic_symbols "${nlsyms_txt}" \
                                    --remove_non_linguistic_symbols true \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/hyp.trn"


                    elif [ "${_type}" = cer ]; then
                        # Tokenize text to char level
                        paste \
                            <(<"${_data}/text" \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type char \
                                    --non_linguistic_symbols "${nlsyms_txt}" \
                                    --remove_non_linguistic_symbols true \
                                    --cleaner "${cleaner}" \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/ref.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text"  \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type char \
                                    --non_linguistic_symbols "${nlsyms_txt}" \
                                    --remove_non_linguistic_symbols true \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/hyp.trn"

                    elif [ "${_type}" = ter ]; then
                        # Tokenize text using BPE
                        paste \
                            <(<"${_data}/text" \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type bpe \
                                    --bpemodel "${bpemodel}" \
                                    --cleaner "${cleaner}" \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/ref.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text" \
                                ${python} -m espnet2.bin.tokenize_text  \
                                    -f 2- --input - --output - \
                                    --token_type bpe \
                                    --bpemodel "${bpemodel}" \
                                    ) \
                            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                                >"${_scoredir}/hyp.trn"

                    fi

                    sclite \
                    ${score_opts} \
                        -r "${_scoredir}/ref.trn" trn \
                        -h "${_scoredir}/hyp.trn" trn \
                        -i rm -o all stdout > "${_scoredir}/result.txt"

                    log "Write ${_type} result in ${_scoredir}/result.txt"
                    grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
                done
            fi
        done

        [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${uasr_exp}"

        # Show results in Markdown syntax
        scripts/utils/show_uasr_result.sh "${uasr_exp}" > "${uasr_exp}"/RESULTS.md
        cat "${uasr_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${uasr_exp}/${uasr_exp##*/}_${inference_uasr_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
        log "Stage 19: Pack model: ${packed_model}"

        _opts=
        if "${use_lm}"; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
            _opts+="--option ${lm_exp}/perplexity_test/ppl "
            _opts+="--option ${lm_exp}/images "
        fi
        if [ "${token_type}" = bpe ]; then
            _opts+="--option ${bpemodel} "
        fi
        if [ "${nlsyms_txt}" != none ]; then
            _opts+="--option ${nlsyms_txt} "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack uasr \
            --uasr_train_config "${uasr_exp}"/config.yaml \
            --uasr_model_file "${uasr_exp}"/"${inference_uasr_model}" \
            ${_opts} \
            --option "${uasr_exp}"/RESULTS.md \
            --option "${uasr_exp}"/RESULTS.md \
            --option "${uasr_exp}"/images \
            --outpath "${packed_model}"
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
        log "Stage 20: Upload model to Zenodo: ${packed_model}"
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
        # /some/where/espnet/egs2/foo/uasr1/ -> foo/uasr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/uasr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${uasr_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${uasr_exp}"/RESULTS.md)</code></pre></li>
<li><strong>uasr config</strong><pre><code>$(cat "${uasr_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${uasr_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
            exit 1
        log "Stage 21: Upload model to HuggingFace: ${hf_repo}"

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
        # /some/where/espnet/egs2/foo/uasr1/ -> foo/uasr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/uasr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=automatic-speech-recognition
        # shellcheck disable=SC2034
        espnet_task=uasr
        # shellcheck disable=SC2034
        task_exp=${uasr_exp}
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
