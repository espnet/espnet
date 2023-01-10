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
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw          # Feature type (raw or fbank_pitch).
audio_format=flac       # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k                  # Sampling rate.
min_wav_duration=0.1    # Minimum duration in second.
max_wav_duration=20     # Maximum duration in second.
use_sid=false           # Whether to use speaker id as the inputs (Need utt2spk in data directory).
feats_extract=fbank     # Type of feature extractor
use_sid=false           # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false           # Whether to use language id as the inputs (Need utt2lang in data directory).
use_discrete_unit=false # Whether to use discrete unit （TODO: jiatong)

# X-vector related
use_xvector=false
xvector_tool=speechbrain
xvector_model=speechbrain/spkrec-ecapa-voxceleb

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
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

# S2ST model related
s2st_tag=        # Suffix to the result dir for s2st model training.
s2st_exp=        # Specify the directory path for S2ST experiment.
               # If this option is specified, s2st_tag is ignored.
s2st_stats_dir=  # Specify the directory path for S2ST statistics.
s2st_config=     # Config for s2st model training.
s2st_args=       # Arguments for s2st model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in s2st config.
pretrained_asr=               # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_s2st=1            # Number of splitting for s2st corpus.
src_lang=es                # source language abbrev. id (e.g., es).
tgt_lang=en                # target language abbrev. id (e.g., en).
use_src_lang=true          # Incorporate ASR loss (use src texts) or not.
use_tgt_lang=true          # Incorporate ST loss (use tgt texts) or not.
write_collected_feats=false  # Whether to dump feature in stats collection (for speed up).

# Upload model related
hf_repo= # Huggingface repositary for model uploading

# Decoding related
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_s2st_model=valid.loss.best.pth # S2ST model path for decoding.
                                      # e.g.
                                      # inference_s2st_model=train.loss.best.pth
                                      # inference_s2st_model=3epoch.pth
                                      # inference_s2st_model=valid.acc.best.pth
                                      # inference_s2st_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
src_bpe_train_text=  # Text file path of bpe training set for source language.
tgt_bpe_train_text=  # Text file path of bpe training set for target language.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
tgt_g2p=none     # g2p method (needed if tgt_token_type=phn).
src_g2p=none     # g2p method (needed if src_token_type=phn).
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
s2st_speech_fold_length=800 # fold_length for speech data during S2ST training.
s2st_text_fold_length=150   # fold_length for text data during S2ST training.

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

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --use_sid          # Whether to use speaker id as the inputs (Need utt2spk in data directory) (default="${use_sid}").
    --feats_extract    # Type of feature extractor (default="${feats_extract}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --use_discrete_unit # Whether to use discrete unit （TODO: jiatong) (default="${use_discrete_unit}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_joint=false       # Whether to use a single bpe system for both source and target languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --src_token_type=bpe      # Tokenization type (char or bpe) for source languages. (default="${src_token_type}").
    --src_nbpe=30             # The number of BPE vocabulary for source language. (default="${src_nbpe}").
    --src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe). (default="${src_bpemode}").
    --src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language. (default="${src_bpe_input_sentence_size}").
    --src_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${src_bpe_nlsyms}").
    --src_bpe_char_cover=1.0  # Character coverage when modeling BPE for source language. (default="${src_bpe_char_cover}").
    --tgt_token_type=bpe      # Tokenization type (char or bpe) for target language. (default="${tgt_token_type}").
    --tgt_nbpe=30             # The number of BPE vocabulary for target language. (default="${tgt_nbpe}").
    --tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language. (default="${tgt_bpemode}").
    --tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language. (default="${tgt_bpe_input_sentence_size}").
    --tgt_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE for target language. (default="${tgt_bpe_nlsyms}").
    --tgt_bpe_char_cover=1.0  # Character coverage when modeling BPE for target language. (default="${tgt_bpe_char_cover}").

    # S2ST model related
    --s2st_tag           # Suffix to the result dir for s2st model training (default="${s2st_tag}").
    --s2st_exp           # Specify the directory path for S2ST experiment.
                       # If this option is specified, s2st_tag is ignored (default="${s2st_exp}").
    --s2st_stats_dir     # Specify the directory path for S2ST statistics (default="${s2st_stats_dir}").
    --s2st_config        # Config for s2st model training (default="${s2st_config}").
    --s2st_args          # Arguments for s2st model training (default="${s2st_args}").
                       # e.g., --s2st_args "--max_epoch 10"
                       # Note that it will overwrite args in s2st config.
    --pretrained_asr=          # Pretrained model to load (default="${pretrained_asr}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type. (default="${feats_normalize}").
    --num_splits_s2st    # Number of splitting for s2st corpus.  (default="${num_splits_s2st}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}").
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}").
    --use_src_lang=    # Incorporate ASR loss (use src texts) or not. (default="${use_src_lang}").
    --use_tgt_lang=    # Incorporate ST loss (use tgt texts) or not. (default="${use_tgt_lang}").
    --write_collected_feats # Whether to dump feature in stats collection for speed up. (default="${write_collected_feats}")

    # Upload model related
    --hf_repo          # Huggingface repositary for model uploading (default="${hf_repo}")

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_s2st_model # S2ST model path for decoding (default="${inference_s2st_model}").
    --vocoder_file        # Vocoder paramemter file (default=${vocoder_file}).
                          # If set to none, Griffin-Lim vocoder will be used.
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --src_bpe_train_text # Text file path of bpe training set for source language.
    --tgt_bpe_train_text # Text file path of bpe training set for target language
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --tgt_g2p           # g2p method for target language (default="${tgt_g2p}").
    --src_g2p           # g2p method for source language (default="${src_g2p}").
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --s2st_speech_fold_length # fold_length for speech data during S2ST training (default="${s2st_speech_fold_length}").
    --s2st_text_fold_length   # fold_length for text data during S2ST training (default="${s2st_text_fold_length}").
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

# Extra files for translation/synthesis process
utt_extra_files="wav.scp.${src_lang} wav.scp.${tgt_lang}"
if [ ${use_src_lang} = true ]; then
    utt_extra_files="${utt_extra_files} text.${src_lang}"
fi
if [ ${use_tgt_lang} = true ]; then
    utt_extra_files="${utt_extra_files} text.${tgt_lang}"
fi


# Use the same text as S2ST for bpe training if not specified.
[ -z "${src_bpe_train_text}" ] && [ $use_src_lang = true ] && src_bpe_train_text="${data_feats}/${train_set}/text.${src_lang}"
[ -z "${tgt_bpe_train_text}" ] && tgt_bpe_train_text="${data_feats}/${train_set}/text.${tgt_lang}"

# Check tokenization type
token_listdir=data/${src_lang}_${tgt_lang}_token_list
# The tgt bpedir is set for all cases when using bpe
tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}"
tgt_bpeprefix="${tgt_bpedir}"/bpe
tgt_bpemodel="${tgt_bpeprefix}".model
tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
tgt_chartoken_list="${token_listdir}"/char/tgt_tokens.txt
tgt_phntoken_list="${token_listdir}"/"phn_${tgt_g2p}"/tgt_tokens.txt
if "${token_joint}"; then
    # if token_joint, the bpe training will use both src_lang and tgt_lang to train a single bpe model
    src_bpedir="${tgt_bpedir}"
    src_bpeprefix="${tgt_bpeprefix}"
    src_bpemodel="${tgt_bpemodel}"
    src_bpetoken_list="${tgt_bpetoken_list}"
    src_chartoken_list="${tgt_chartoken_list}"
    src_phntoken_list="${tgt_phntoken_list}"
else
    src_bpedir="${token_listdir}/src_bpe_${src_bpemode}${src_nbpe}"
    src_bpeprefix="${src_bpedir}"/bpe
    src_bpemodel="${src_bpeprefix}".model
    src_bpetoken_list="${src_bpedir}"/tokens.txt
    src_chartoken_list="${token_listdir}"/char/src_tokens.txt
    src_phntoken_list="${token_listdir}"/"phn_${src_g2p}"/src_tokens.txt
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
tgt_wordtoken_list="${token_listdir}"/word/tgt_tokens.txt
if "${token_joint}"; then
    src_wordtoken_list="${tgt_wordtoken_list}"
else
    src_wordtoken_list="${token_listdir}"/word/src_tokens.txt
fi

# Set token types for src and tgt langs
if [ $use_src_lang = false ]; then
    src_token_type=none
    src_token_list=none
elif [ "${src_token_type}" = bpe ]; then
    src_token_list="${src_bpetoken_list}"
elif [ "${src_token_type}" = char ]; then
    src_token_list="${src_chartoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = word ]; then
    src_token_list="${src_wordtoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = phn ]; then
    src_token_list="${src_phntoken_list}"
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
elif [ "${tgt_token_type}" = phn ]; then
    src_token_list="${tgt_phntoken_list}"
    src_bpemodel=none
else
    log "Error: not supported --tgt_token_type '${tgt_token_type}'"
    exit 2
fi


# Set tag for naming of model directory
if [ -z "${s2st_tag}" ]; then
    if [ -n "${s2st_config}" ]; then
        s2st_tag="$(basename "${s2st_config}" .yaml)_${feats_type}_${feats_extract}"
    else
        s2st_tag="train_${feats_type}_${feats_extract}"
    fi
    s2st_tag+="_${src_lang}_${tgt_lang}"
    if [ "${tgt_token_type}" = bpe ]; then
        s2st_tag+="${tgt_nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${s2st_args}" ]; then
        s2st_tag+="$(echo "${s2st_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${s2st_stats_dir}" ]; then
    s2st_stats_dir="${expdir}/s2st_stats_${feats_type}_${src_lang}_${tgt_lang}"
    if [ "${use_tgt_lang}" = true ] && [ "${tgt_token_type}" = bpe ]; then
        s2st_stats_dir+="_bpe${tgt_nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${s2st_exp}" ]; then
    s2st_exp="${expdir}/s2st_${s2st_tag}"
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
    inference_tag+="_s2st_model_$(echo "${inference_s2st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    # NOTE(jiatong): we may add speed perturbation?

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

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

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls data/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                echo "${expand_utt_extra_files}"
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp.${src_lang},wav.scp,wav.scp.${tgt_lang},reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    # Note(jiatong): we just consider the case for input speech only for now
                    _opts+="--segments data/${dset}/segments "
                fi

                log "Format target wav.scp"
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" --suffix ".${tgt_lang}" \
                    --out_filename "wav.scp.${tgt_lang}" \
                    "data/${dset}/wav.scp.${tgt_lang}" "${data_feats}${_suf}/${dset}"
                
                log "Format source wav.scp"
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" --suffix ".${src_lang}" \
                    --out_filename "wav.scp.${src_lang}" ${_opts} \
                    "data/${dset}/wav.scp.${src_lang}" "${data_feats}${_suf}/${dset}"
                ln -sf "wav.scp.${src_lang}" "${data_feats}${_suf}/${dset}/wav.scp"

                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
            # Assuming you don't have wav.scp, but feats.scp is created by local/data.sh instead.
            # TODO(jiatong): add process scripts for extracted features

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi

        # Extract X-vector
        if "${use_xvector}"; then
            if [ "${xvector_tool}" = "kaldi" ]; then
                log "Stage 2+: Extract X-vector: data/ -> ${dumpdir}/xvector (Require Kaldi)"
                # Download X-vector pretrained model
                xvector_exp=${expdir}/xvector_nnet_1a
                if [ ! -e "${xvector_exp}" ]; then
                    log "X-vector model does not exist. Download pre-trained model."
                    wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
                    tar xvf 0008_sitw_v2_1a.tar.gz
                    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
                    mv 0008_sitw_v2_1a/exp/xvector_nnet_1a "${xvector_exp}"
                    rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
                fi

                # Generate the MFCC features, VAD decision, and X-vector
                for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                    if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                        _suf="/org"
                    else
                        _suf=""
                    fi
                    # 1. Copy datadir and resample to 16k
                    utils/copy_data_dir.sh  --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}" "${dumpdir}/mfcc/${dset}"
                    utils/data/resample_data_dir.sh 16000 "${dumpdir}/mfcc/${dset}"

                    # 2. Extract mfcc features
                    _nj=$(min "${nj}" "$(<${dumpdir}/mfcc/${dset}/utt2spk wc -l)")
                    steps/make_mfcc.sh --nj "${_nj}" --cmd "${train_cmd}" \
                        --write-utt2num-frames true \
                        --mfcc-config conf/mfcc.conf \
                        "${dumpdir}/mfcc/${dset}"
                    utils/fix_data_dir.sh "${dumpdir}/mfcc/${dset}"

                    # 3. Compute VAD decision
                    _nj=$(min "${nj}" "$(<${dumpdir}/mfcc/${dset}/spk2utt wc -l)")
                    sid/compute_vad_decision.sh --nj ${_nj} --cmd "${train_cmd}" \
                        --vad-config conf/vad.conf \
                        "${dumpdir}/mfcc/${dset}"
                    utils/fix_data_dir.sh "${dumpdir}/mfcc/${dset}"

                    # 4. Extract X-vector
                    sid/nnet3/xvector/extract_xvectors.sh --nj "${_nj}" --cmd "${train_cmd}" \
                        "${xvector_exp}" \
                        "${dumpdir}/mfcc/${dset}" \
                        "${dumpdir}/xvector/${dset}"

                    # 5. Filter scp
                    # NOTE(kan-bayashi): Since sometimes mfcc or x-vector extraction is failed,
                    #   the number of utts will be different from the original features (raw or fbank).
                    #   To avoid this mismatch, perform filtering of the original feature scp here.
                    cp "${data_feats}${_suf}/${dset}"/wav.{scp.${src_lang},scp.${src_lang}.bak}
                    <"${data_feats}${_suf}/${dset}/wav.scp.${src_lang}.bak" \
                        utils/filter_scp.pl "${dumpdir}/xvector/${dset}/xvector.scp" \
                        >"${data_feats}${_suf}/${dset}/wav.scp.${src_lang}"
                    utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"
                done
            else
                # Assume that others toolkits are python-based
                log "Stage 2+: Extract X-vector: data/ -> ${dumpdir}/xvector using python toolkits"
                for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                    if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                        _suf="/org"
                    else
                        _suf=""
                    fi
                    pyscripts/utils/extract_xvectors.py \
                        --pretrained_model ${xvector_model} \
                        --toolkit ${xvector_tool} \
                        ${data_feats}${_suf}/${dset} \
                        ${dumpdir}/xvector/${dset}
                done
            fi
        fi

        # Prepare spk id input
        if "${use_sid}"; then
            log "Stage 2+: Prepare speaker id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make spk2sid
                    # NOTE(kan-bayashi): 0 is reserved for unknown speakers
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/spk2sid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2spk" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/spk2sid"
                fi
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/spk2sid" \
                    "${data_feats}${_suf}/${dset}/utt2spk" \
                    > "${data_feats}${_suf}/${dset}/utt2sid"
            done
        fi

        # Prepare lang id input
        if "${use_lid}"; then
            log "Stage 2+: Prepare lang id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make lang2lid
                    # NOTE(kan-bayashi): 0 is reserved for unknown languages
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/lang2lid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2lang" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/lang2lid"
                fi
                # NOTE(kan-bayashi): We can reuse the same script for making utt2sid
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/lang2lid" \
                    "${data_feats}${_suf}/${dset}/utt2lang" \
                    > "${data_feats}${_suf}/${dset}/utt2lid"
            done
        fi

    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
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

                for lang in "${src_lang}" "${tgt_lang}"; do
                    # utt2num_samples is created by format_wav_scp.sh
                    <"${data_feats}/org/${dset}/utt2num_samples.${lang}" \
                        awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                            '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                            >"${data_feats}/${dset}/utt2num_samples.${lang}"
                    <"${data_feats}/org/${dset}/wav.scp.${lang}" \
                        utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples.${lang}"  \
                        >"${data_feats}/${dset}/wav.scp.${lang}"
                done
            else
                log "Not supported feature type ${_feats_type}."
                exit 2
            fi

            # Remove empty text
            for utt_extra_file in ${utt_extra_files}; do
                <"${data_feats}/org/${dset}/${utt_extra_file}" \
                    awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/${dset}/${utt_extra_file}"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"

            # NOTE(jiatong): some extra treatment for extra files, including sorting and duplication remove
            for utt_extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py ${data_feats}/${dset}/${utt_extra_file} \
                    > ${data_feats}/${dset}/${utt_extra_file}.tmp
                mv ${data_feats}/${dset}/${utt_extra_file}.tmp ${data_feats}/${dset}/${utt_extra_file}
		        sort -o ${data_feats}/${dset}/${utt_extra_file} ${data_feats}/${dset}/${utt_extra_file}
            done
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        # Combine source and target texts when using joint tokenization
        if "${token_joint}"; then
            log "Merge src and target data if joint BPE"

            cat $tgt_bpe_train_text > ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
            [ ! -z "${src_bpe_train_text}" ] && cat ${src_bpe_train_text} >> ${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}
            # Set the new text as the target text
            tgt_bpe_train_text="${data_feats}/${train_set}/text.${src_lang}_${tgt_lang}"
        fi

        # First generate tgt lang
        if [ "${tgt_token_type}" = bpe ]; then
            log "Stage 4a: Generate token_list from ${tgt_bpe_train_text} using BPE for tgt_lang"

            mkdir -p "${tgt_bpedir}"
            # shellcheck disable=SC2002
            cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${tgt_bpedir}"/train.txt

            if [ -n "${tgt_bpe_nlsyms}" ]; then
                _opts_spm="--user_defined_symbols=${tgt_bpe_nlsyms}"
            else
                _opts_spm=""
            fi

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

        elif [ "${tgt_token_type}" = char ] || [ "${tgt_token_type}" = word ] || [ "${tgt_token_type}" = phn ]; then
            log "Stage 5a: Generate character level token_list from ${tgt_bpe_train_text}  for tgt_lang"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # shellcheck disable=SC2002
            cat ${tgt_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for S2ST and also used as ignore-index in the other task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${tgt_token_type}" \
                --input "${data_feats}/token_train.txt" --output "${tgt_token_list}" ${_opts} \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${tgt_g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"

        else
            log "Error: not supported --token_type '${tgt_token_type}'"
            exit 2
        fi

        # Then generate src lang
        if "${token_joint}"; then
            log "Stage 4b: Skip separate token construction for src_lang when setting ${token_joint} as true"
        elif [ $use_src_lang = true ]; then
            if [ "${src_token_type}" = bpe ]; then
                log "Stage 4b: Generate token_list from ${src_bpe_train_text} using BPE for src_lang"

                mkdir -p "${src_bpedir}"
                # shellcheck disable=SC2002
                cat ${src_bpe_train_text} | cut -f 2- -d" "  > "${src_bpedir}"/train.txt

                if [ -n "${src_bpe_nlsyms}" ]; then
                    _opts_spm="--user_defined_symbols=${src_bpe_nlsyms}"
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

            elif [ "${src_token_type}" = char ] || [ "${src_token_type}" = word ] || [ "${tgt_token_type}" = phn ]; then
                log "Stage 4b: Generate character level token_list from ${src_bpe_train_text}  for src_lang"

                _opts="--non_linguistic_symbols ${nlsyms_txt}"

                # shellcheck disable=SC2002
                cat ${src_bpe_train_text} | cut -f 2- -d" "  > "${data_feats}"/token_train.txt

                # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
                # 0 is reserved for CTC-blank for S2ST and also used as ignore-index in the other task
                ${python} -m espnet2.bin.tokenize_text  \
                    --token_type "${src_token_type}" \
                    --input "${data_feats}/token_train.txt" --output "${src_token_list}" ${_opts} \
                    --field 2- \
                    --cleaner "${cleaner}" \
                    --g2p "${src_g2p}" \
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
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _s2st_train_dir="${data_feats}/${train_set}"
        _s2st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: S2ST collect stats: train_set=${_s2st_train_dir}, valid_set=${_s2st_valid_dir}"

        _opts=
        if [ -n "${s2st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.s2st_train --print_config --optim adam
            _opts+="--config ${s2st_config} "
        fi

        _feats_type="$(<${_s2st_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _tgt_scp=wav.scp.${tgt_lang}
            _src_scp=wav.scp.${src_lang}
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            log "Error: not supported feature type '${_feats_type}'"
            exit 2
        fi

        # 1. Split the key file
        _logdir="${s2st_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_s2st_train_dir}/${_tgt_scp} wc -l)" "$(<${_s2st_valid_dir}/${_tgt_scp} wc -l)")

        key_file="${_s2st_train_dir}/${_tgt_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_s2st_valid_dir}/${_tgt_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${s2st_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${s2st_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${s2st_stats_dir}/run.sh"; chmod +x "${s2st_stats_dir}/run.sh"

        # 3. Submit jobs
        log "S2ST collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        if [ $use_tgt_lang = true ]; then
            _opts+="--tgt_bpemodel ${tgt_bpemodel} "
            _opts+="--tgt_token_type ${tgt_token_type} "
            _opts+="--tgt_token_list ${tgt_token_list} "
            _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/text.${tgt_lang},tgt_text,text "
            _opts+="--valid_data_path_and_name_and_type ${_s2st_valid_dir}/text.${tgt_lang},tgt_text,text "
        fi

        if [ $use_src_lang = true ]; then
            _opts+="--src_bpemodel ${src_bpemodel} "
            _opts+="--src_token_type ${src_token_type} "
            _opts+="--src_token_list ${src_token_list} "
            _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/text.${src_lang},src_text,text "
            _opts+="--valid_data_path_and_name_and_type ${_s2st_valid_dir}/text.${src_lang},src_text,text "
        fi
        # TODO(jiatong): fix different bpe model
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.s2st_train \
                --collect_stats true \
                --use_preprocessor true \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --tgt_g2p "${tgt_g2p}" \
                --src_g2p "${src_g2p}" \
                --train_data_path_and_name_and_type "${_s2st_train_dir}/${_src_scp},src_speech,${_type}" \
                --train_data_path_and_name_and_type "${_s2st_train_dir}/${_tgt_scp},tgt_speech,${_type}" \
                --valid_data_path_and_name_and_type "${_s2st_valid_dir}/${_src_scp},src_speech,${_type}" \
                --valid_data_path_and_name_and_type "${_s2st_valid_dir}/${_tgt_scp},tgt_speech,${_type}" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${s2st_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${s2st_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        if [ ${use_tgt_lang} = true ]; then
            <"${s2st_stats_dir}/train/tgt_text_shape" \
                awk -v N="$(<${tgt_token_list} wc -l)" '{ print $0 "," N }' \
                >"${s2st_stats_dir}/train/tgt_text_shape.${tgt_token_type}"

            <"${s2st_stats_dir}/valid/tgt_text_shape" \
                awk -v N="$(<${tgt_token_list} wc -l)" '{ print $0 "," N }' \
                >"${s2st_stats_dir}/valid/tgt_text_shape.${tgt_token_type}"
        fi

        
        if [ ${use_src_lang} = true ]; then
            <"${s2st_stats_dir}/train/src_text_shape" \
                awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
                >"${s2st_stats_dir}/train/src_text_shape.${src_token_type}"

            <"${s2st_stats_dir}/valid/src_text_shape" \
                awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
                >"${s2st_stats_dir}/valid/src_text_shape.${src_token_type}"
        fi
    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _s2st_train_dir="${data_feats}/${train_set}"
        _s2st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: S2ST Training: train_set=${_s2st_train_dir}, valid_set=${_s2st_valid_dir}"

        _opts=
        if [ -n "${s2st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.s2st_train --print_config --optim adam
            _opts+="--config ${s2st_config} "
        fi

        _feats_type="$(<${_s2st_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _tgt_scp=wav.scp.${tgt_lang}
            _src_scp=wav.scp.${src_lang}
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((s2st_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${s2st_speech_fold_length}"
            _input_size="$(<${_s2st_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi

        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/xvector/${train_set}"
            _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        if "${use_sid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
        fi

        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--src_normalize=global_mvn --src_normalize_conf stats_file=${s2st_stats_dir}/train/src_feats_stats.npz "
            _opts+="--tgt_normalize=global_mvn --tgt_normalize_conf stats_file=${s2st_stats_dir}/train/tgt_feats_stats.npz "
        fi

        _num_splits_opts=
        if [ ${use_tgt_lang} = true ]; then
            _num_splits_opts+="${_s2st_train_dir}/text.${tgt_lang} " 
            _num_splits_opts+="${s2st_stats_dir}/train/tgt_text_shape.${tgt_token_type} " 
        fi
        if [ ${use_src_lang} = true ]; then
            _num_splits_opts+="${_s2st_train_dir}/text.${src_lang} " 
            _num_splits_opts+="${s2st_stats_dir}/train/src_text_shape.${src_token_type} " 
        fi

        if [ "${num_splits_s2st}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${s2st_stats_dir}/splits${num_splits_s2st}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_s2st_train_dir}/${_tgt_scp}" \
                      "${s2st_stats_dir}/train/tgt_speech_shape" \
                      "${_s2st_train_dir}/${_src_scp}" \
                      "${s2st_stats_dir}/train/src_speech_shape" \
                      $_num_splits_opts \
                  --num_splits "${num_splits_s2st}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_tgt_scp},tgt_speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_src_scp},src_speech,${_type} "
            _opts+="--train_shape_file ${_split_dir}/tgt_speech_shape "
            _opts+="--train_shape_file ${_split_dir}/src_speech_shape "
            _opts+="--multiple_iterator true "
            if [ ${use_src_lang} = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_lang},src_text,text "
                _opts+="--train_shape_file ${_split_dir}/src_text_shape.${src_token_type} "
            fi 
            if [ ${use_tgt_lang} = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${tgt_lang},tgt_text,text "
                _opts+="--train_shape_file ${_split_dir}/tgt_text_shape.${tgt_token_type} "
            fi 
        else
            _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/${_tgt_scp},tgt_speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/${_src_scp},src_speech,${_type} "
            _opts+="--train_shape_file ${s2st_stats_dir}/train/tgt_speech_shape "
            _opts+="--train_shape_file ${s2st_stats_dir}/train/src_speech_shape "
            if [ $use_src_lang = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/text.${src_lang},src_text,text "
                _opts+="--train_shape_file ${s2st_stats_dir}/train/src_text_shape.${src_token_type} "
            fi
            if [ ${use_tgt_lang} = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_s2st_train_dir}/text.${tgt_lang},tgt_text,text "
                _opts+="--train_shape_file ${s2st_stats_dir}/train/tgt_text_shape.${tgt_token_type} "
            fi
        fi

        log "Generate '${s2st_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${s2st_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${s2st_exp}/run.sh"; chmod +x "${s2st_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "S2ST training started... log: '${s2st_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${s2st_exp})"
        else
            jobname="${s2st_exp}/train.log"
        fi

        if [ ${use_tgt_lang} = true ]; then
            _opts+="--tgt_token_type ${tgt_token_type} "
            _opts+="--tgt_token_list ${tgt_token_list} "
            _opts+="--tgt_bpemodel ${tgt_bpemodel} "
            _opts+="--valid_data_path_and_name_and_type ${_s2st_valid_dir}/text.${tgt_lang},tgt_text,text " 
            _opts+="--valid_shape_file ${s2st_stats_dir}/valid/tgt_text_shape.${tgt_token_type} " 
            _opts+="--fold_length ${s2st_text_fold_length} "
        fi
        if [ ${use_src_lang} = true ]; then
            _opts+="--src_token_type ${src_token_type} "
            _opts+="--src_token_list ${src_token_list} "
            _opts+="--src_bpemodel ${src_bpemodel} "
            _opts+="--valid_data_path_and_name_and_type ${_s2st_valid_dir}/text.${src_lang},src_text,text " 
            _opts+="--valid_shape_file ${s2st_stats_dir}/valid/src_text_shape.${src_token_type} " 
            _opts+="--fold_length ${s2st_text_fold_length} "
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${s2st_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${s2st_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.s2st_train \
                --use_preprocessor true \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --tgt_g2p "${tgt_g2p}" \
                --src_g2p "${src_g2p}" \
                --valid_data_path_and_name_and_type "${_s2st_valid_dir}/${_src_scp},src_speech,${_type}" \
                --valid_data_path_and_name_and_type "${_s2st_valid_dir}/${_tgt_scp},tgt_speech,${_type}" \
                --valid_shape_file "${s2st_stats_dir}/valid/src_speech_shape" \
                --valid_shape_file "${s2st_stats_dir}/valid/tgt_speech_shape" \
                --resume true \
                --init_param ${pretrained_asr} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${_fold_length}" \
                --fold_length "${s2st_text_fold_length}" \
                --output_dir "${s2st_exp}" \
                ${_opts} ${s2st_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    s2st_exp="${expdir}/${download_model}"
    mkdir -p "${s2st_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${s2st_exp}/config.txt"

    # Get the path of each file
    _s2st_model_file=$(<"${s2st_exp}/config.txt" sed -e "s/.*'s2st_model_file': '\([^']*\)'.*$/\1/")
    _s2st_train_config=$(<"${s2st_exp}/config.txt" sed -e "s/.*'s2st_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_s2st_model_file}" "${s2st_exp}"
    ln -sf "${_s2st_train_config}" "${s2st_exp}"
    inference_s2st_model=$(basename "${_s2st_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Decoding: training_dir=${s2st_exp}"

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
        log "Generate '${s2st_exp}/${inference_tag}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${s2st_exp}/${inference_tag}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${s2st_exp}/${inference_tag}/run.sh"; chmod +x "${s2st_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${s2st_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _src_scp=wav.scp.${src_lang}
                _tgt_scp=wav.scp.${tgt_lang}
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _src_scp=feats.scp.${src_lang}
                _tgt_scp=feats.scp.${tgt_lang}
                _type=kaldi_ark
            fi

            _ex_opts=""

            # Add X-vector to the inputs if needed
            if "${use_xvector}"; then
                _xvector_dir="${dumpdir}/xvector/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_xvector_dir}/xvector.scp,spembs,kaldi_ark "
            fi

            # Add spekaer ID to the inputs if needed
            if "${use_sid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2sid,sids,text_int "
            fi

            # Add language ID to the inputs if needed
            if "${use_lid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2lid,lids,text_int "
            fi

            # 1. Split the key file
            key_file=${_data}/${_src_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/s2st_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/s2st_inference.JOB.log \
                ${python} -m espnet2.bin.s2st_inference \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_src_scp},src_speech,${_type}" \
                    --data_path_and_name_and_type "${_data}/${_tgt_scp},tgt_speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${s2st_exp}"/config.yaml \
                    --model_file "${s2st_exp}"/"${inference_s2st_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    --vocoder_file "${vocoder_file}" \
                    ${_opts} ${_ex_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/s2st_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            if [ -e "${_logdir}/output.${_nj}/norm" ]; then
                mkdir -p "${_dir}"/norm
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/norm/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/denorm" ]; then
                mkdir -p "${_dir}"/denorm
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/denorm/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/speech_shape" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/speech_shape/speech_shape"
                done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
            fi
            if [ -e "${_logdir}/output.${_nj}/wav" ]; then
                mkdir -p "${_dir}"/wav
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                    rm -rf "${_logdir}/output.${i}"/wav
                done
                find "${_dir}/wav" -name "*.wav" | while read -r line; do
                    echo "$(basename "${line}" .wav) ${line}"
                done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
                mkdir -p "${_dir}"/att_ws
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                    rm -rf "${_logdir}/output.${i}"/att_ws
                done
            fi
            if [ -e "${_logdir}/output.${_nj}/durations" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/durations/durations"
                done | LC_ALL=C sort -k1 > "${_dir}/durations"
            fi
            if [ -e "${_logdir}/output.${_nj}/focus_rates" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/focus_rates/focus_rates"
                done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
            fi
            if [ -e "${_logdir}/output.${_nj}/probs" ]; then
                mkdir -p "${_dir}"/probs
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
                    rm -rf "${_logdir}/output.${i}"/probs
                done
            fi
        done
    fi

    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 13: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${s2st_exp}/${inference_tag}/${dset}"

            # TODO(jiatong): add asr scoring and inference

            _scoredir="${_dir}/score_bleu"
            mkdir -p "${_scoredir}"

            paste \
                <(<"${_data}/text.${tgt_lang}" \
                    ${python} -m espnet2.bin.tokenize_text  \
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
                <(<"${_dir}/text"  \
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
            detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
            detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

            # rotate result files
            pyscripts/utils/rotate_logfile.py ${_scoredir}/result.lc.txt


            # detokenize & remove punctuation except apostrophe
            remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
            remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"
            echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
            sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                      -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                      -m bleu chrf ter \
                      >> ${_scoredir}/result.lc.txt
            log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"

            # process multi-references cases
            multi_references=$(ls "${_data}/text.${tgt_lang}".* || echo "")
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
                    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn.${ref_idx}" > "${_scoredir}/ref.trn.detok.${ref_idx}"
                    remove_punctuation.pl < "${_scoredir}/ref.trn.detok.${ref_idx}" > "${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                    case_sensitive_refs="${case_sensitive_refs} ${_scoredir}/ref.trn.detok.${ref_idx}"
                    case_insensitive_refs="${case_insensitive_refs} ${_scoredir}/ref.trn.detok.lc.rm.${ref_idx}"
                done


                echo "Case insensitive BLEU result (multi-references)" >> ${_scoredir}/result.lc.txt
                sacrebleu -lc ${case_insensitive_refs} \
                    -i ${_scoredir}/hyp.trn.detok.lc.rm -m bleu chrf ter \
                    >> ${_scoredir}/result.lc.txt
                log "Write a case-insensitve BLEU (multi-reference) result in ${_scoredir}/result.lc.txt"
            fi
        done

        # Show results in Markdown syntax
        scripts/utils/show_translation_result.sh --case $tgt_case "${s2st_exp}" > "${s2st_exp}"/RESULTS.md
        cat "${s2st_exp}"/RESULTS.md
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${s2st_exp}/${s2st_exp##*/}_${inference_s2st_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Pack model: ${packed_model}"

        _opts=
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${s2st_stats_dir}/train/feats_stats.npz "
        fi
        if [ "${tgt_token_type}" = bpe ]; then
            _opts+="--option ${tgt_bpemodel} "
            _opts+="--option ${src_bpemodel} "
        fi
        if [ "${nlsyms_txt}" != none ]; then
            _opts+="--option ${nlsyms_txt} "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack s2st \
            --s2st_train_config "${s2st_exp}"/config.yaml \
            --s2st_model_file "${s2st_exp}"/"${inference_s2st_model}" \
            ${_opts} \
            --option "${s2st_exp}"/RESULTS.md \
            --option "${s2st_exp}"/RESULTS.md \
            --option "${s2st_exp}"/images \
            --outpath "${packed_model}"
    fi


    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        log "Stage 15: Upload model to Zenodo: ${packed_model}"

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
        # /some/where/espnet/egs2/foo/s2st1/ -> foo/s2st1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/s2st1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${s2st_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${s2st_exp}"/RESULTS.md)</code></pre></li>
<li><strong>S2ST config</strong><pre><code>$(cat "${s2st_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${src_lang}_${tgt_lang}" \
            --description_file "${s2st_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 16: Upload model to HuggingFace: ${hf_repo}"

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
        hf_task=speech-translation
        # shellcheck disable=SC2034
        espnet_task=S2ST
        # shellcheck disable=SC2034
        task_exp=${s2st_exp}
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
