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
skip_upload=true     # Skip packing and uploading stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw         # Feature type (raw or fbank_pitch).
audio_format=flac      # Audio format (only in feats_type=raw).
fs=16k                 # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

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

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment. If this option is specified, lm_tag is ignored.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# Joint model related
joint_tag=    # Suffix to the result dir for enhancement model training.
joint_config= # Config for ehancement + ASR model joint training.
joint_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
              # Note that it will overwrite args in enhancement config.
joint_exp=    # Specify the direcotry path for ASR experiment. If this option is specified, joint_tag is ignored.

# Enhancement model related
spk_num=2
noise_type_num=1
dereverb_ref_num=1

# ASR model related
feats_normalize=global_mvn  # Normalizaton layer type
num_splits_asr=1            # Number of splitting for lm corpus

# Training data related
use_signal_ref=true
use_signal_dereverb_ref=false
use_signal_noise_ref=false

# Evaluation related
scoring_protocol="STOI SDR SAR SIR"
ref_channel=0

# Decoding related
decode_lm=valid.loss.best.pth               # Language modle path for decoding.
decode_tag=                                 # Suffix to the result dir for decoding.
decode_config=                              # Config for decoding.
decode_args="--normalize_output_wav true "  # Arguments for decoding, e.g., "--lm_weight 0.1".
                                            # Note that it will overwrite args in decode config.
decode_joint_model=valid.acc.ave.pth        # ASR model path for decoding.
                                            # e.g.
                                            # decode_joint_model=train.loss.best.pth
                                            # decode_joint_model=3epoch.pth
                                            # decode_joint_model=valid.acc.best.pth
                                            # decode_joint_model=valid.loss.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
train_aux_set=   # Name of auxiliary training set. (Single Speaker)
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training

srctexts=        # Used for the training of BPE and LM and the creation of a vocabulary list.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus
asr_speech_fold_length=800 # fold_length for speech data during ASR training
asr_text_fold_length=150   # fold_length for text data during ASR training
lm_fold_length=150         # fold_length for LM training

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference  # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type     # Feature type (only support raw currently).
    --audio_format   # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs             # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos=                # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the direcotry path for LM experiment. If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training, e.g., "--max_epoch 10" (default="${lm_args}").
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm=1   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # Joint model related
    --joint_tag    # Suffix to the result dir for enhancement model training (default="${joint_tag}").
    --joint_config # Config for enhancement + ASR model joint training (default="${joint_config}").
    --joint_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${joint_args}").
                   # Note that it will overwrite args in enhancement config.
    --joint_exp    # Specify the direcotry path for ASR experiment. If this option is specified, joint_tag is ignored.

    # Enhancement model related
    --spk_num          # Number of speakers in the input audio (default="${spk_num}")
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")

    # ASR model related
    --feats_normalize    # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr=1   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --decode_tag       # Suffix to the result dir for decoding (default="${decode_tag}").
    --decode_config    # Config for decoding (default="${decode_config}").
    --decode_args      # Arguments for decoding, e.g., "--lm_weight 0.1" (default="${decode_args}").
                       # Note that it will overwrite args in decode config.
    --decode_lm        # Language modle path for decoding (default="${decode_lm}").
    --decode_joint_model # Enhancement model path for inference (default="${decode_joint_model}").


    # Training data related
    --use_signal_ref          # Whether or not to use signal references to compute signal-level losses
                         (default="${use_signal_ref}")
    --use_signal_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_signal_dereverb_ref}")
    --use_signal_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_signal_noise_ref}")


    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --train_aux_set=     # Name of auxilary sets (default="${train_aux_set}")
    --valid_set=    # Name of validation set used for monitoring/tuning network training (required).
    --test_sets=    # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified (required).
    --srctexts      # Used for the training of BPE and LM and the creation of a vocabulary list (required).
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang              # The language type of corpus (default=${lang}).
    --asr_speech_fold_length # fold_length for speech data during ASR training  (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training  (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training  (default="${lm_fold_length}").
    --enh_speech_fold_length # fold_length for speech data during enhancement training  (default="${enh_speech_fold_length}").
EOF
)

log "$0 $*"
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
[ -z "${train_aux_set}" ] && { log "${help_message}"; log "Warning: --train_aux_set is empty. No auxiliary training data is provided."; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
fi


if [ ${spk_num} -le 1 ]; then
    # Use the same text as ASR for lm training if not specified.
    [ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
    # Use the text of the 1st evaldir if lm_test is not specified
    [ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"
fi

# Check tokenization type
token_listdir=data/token_list
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
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
if [ -z "${joint_tag}" ]; then
    if [ -n "${joint_config}" ]; then
        joint_tag="$(basename "${joint_config}" .yaml)_${feats_type}_${token_type}"
    else
        joint_tag="train_${feats_type}_${token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${joint_args}" ]; then
        joint_tag+="$(echo "${joint_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)_${lm_token_type}"
    else
        lm_tag="train_${lm_token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${decode_tag}" ]; then
    if [ -n "${decode_config}" ]; then
        decode_tag="$(basename "${decode_config}" .yaml)"
    else
        decode_tag=decode
    fi
    # Add overwritten arg's info
    #if [ -n "${decode_args}" ]; then
    #    decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    #fi
    if "${use_lm}"; then
        decode_tag+="_lm_${lm_tag}_$(echo "${decode_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    decode_tag+="_asr_model_$(echo "${decode_joint_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
joint_stats_dir="${expdir}/joint_stats_${feats_type}_${fs}"
if [ -n "${speed_perturb_factors}" ]; then
    joint_stats_dir="${joint_stats_dir}_sp"
fi
if [ -n "${train_aux_set}" ]; then
    joint_stats_dir="${joint_stats_dir}_aux"
fi
lm_stats_dir="${expdir}/lm_stats"

# The directory used for training commands
if [ -z "${joint_exp}" ]; then
    joint_exp="${expdir}/joint_${joint_tag}"
fi
if [ -n "${speed_perturb_factors}" ]; then
    joint_exp="${joint_exp}_sp"
fi
if [ -n "${train_aux_set}" ] && [[ ! "${joint_exp}" =~ "_aux" ]]; then
    joint_exp="${joint_exp}_aux"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if ! $use_signal_dereverb_ref && [ -n "${speed_perturb_factors}" ]; then
            log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

            _scp_list="wav.scp "
            for i in $(seq ${spk_num}); do
                _scp_list+="spk${i}.scp "
            done

            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    scripts/utils/perturb_enh_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
                    _dirs+="data/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="data/${train_set} "
                fi
            done
            utils/combine_data.sh --extra-files "${_scp_list}" "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}/org/"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${train_aux_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [[ "${dset}" = "${train_aux_set}" ]] || \
                    [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
                # TODO(Jing): should be more precise.
                cp data/"${dset}"/text* "${data_feats}${_suf}/${dset}"
                if [ -f data/"${dset}"/utt2category ]; then
                    cp data/"${dset}"/utt2category "${data_feats}${_suf}/${dset}"
                else
                    log "No utt2category found."
                fi
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi

                _spk_list=" "
                if ${use_signal_ref}; then
                    for i in $(seq ${spk_num}); do
                        _spk_list+="spk${i} "
                    done
                    if $use_signal_noise_ref; then
                        # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                        _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                    fi
                    if $use_signal_dereverb_ref; then
                        # references for dereverberation
                        _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                    fi
                fi

                for spk in ${_spk_list} "wav" ; do
                    # shellcheck disable=SC2086
                    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                        --out-filename "${spk}.scp" \
                        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                        "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                        "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

                done
                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            done
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${train_aux_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            if [ -f ${data_feats}/org/${dset}/utt2category ]; then
                cp ${data_feats}/org/${dset}/utt2category "${data_feats}/${dset}/utt2category"
                utt_extra_files="utt2category "
            else
                log "No utt2category found."
                utt_extra_files=""
            fi
            if [ -f ${data_feats}/org/${dset}/text_spk1 ]; then
                for i in $(seq ${spk_num}); do
                    cp "${data_feats}/org/${dset}/text_spk${i}" "${data_feats}/${dset}/text_spk${i}"
                    utt_extra_files=${utt_extra_files}"text_spk${i} "
                done
            fi

            _spk_list=" "
            if ${use_signal_ref}; then
                for i in $(seq ${spk_num}); do
                    _spk_list+="spk${i} "
                done
                if ${use_signal_noise_ref}; then
                    # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                    _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                fi
                if ${use_signal_dereverb_ref}; then
                    # references for dereverberation
                    _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                fi
            fi

            # TODO(Jing): conduct single-spk aux sets independetly
            # This part should remove the spk1,spk2... in the spk_list for single-spk set

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

                filter_list="wav"
                if ${use_signal_ref}; then
                    for spk in ${_spk_list};do
                        filter_list=${spk}" "${filter_list}
                        cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
                    done
                fi

                for file in ${filter_list}; do
                    <"${data_feats}/org/${dset}/${file}.scp" \
                        utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                        >"${data_feats}/${dset}/${file}.scp"
                done
            fi

            for i in $(seq ${spk_num}); do
                <"${data_feats}/org/${dset}/text_spk${i}" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/text_spk${i}"
            done
            # Remove empty text
            for i in $(seq ${spk_num}); do
                <"${data_feats}/org/${dset}/text_spk${i}" \
                    awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text_spk${i}"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"
        done

        if [ -n "${train_aux_set}" ]; then
            combined_train_set=${train_set}"_"${train_aux_set}
            log "Combine the aux set with the train set into new set: ${data_feats}/${combined_train_set}."
            if [ -d ${data_feats}/${combined_train_set} ]; then
                rm -r ${data_feats:?"data_feats is empty"}/${combined_train_set} 2>/dev/null
            fi
            ./utils/combine_data.sh --skip-fix true --extra-files "spk1.scp spk2.scp text_spk1 text_spk2 utt2category utt2num_samples" \
             "${data_feats}/${combined_train_set}" "${data_feats}/${train_set}" "${data_feats}/${train_aux_set}"
            
            cp ${data_feats}/${dset}/feats_type ${data_feats}/${combined_train_set}/feats_type
        else
            log "Train set is the same as aux sets."
        fi
    fi

    if [ -n "${train_aux_set}" ]; then
        train_set=${train_set}"_"${train_aux_set}
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = bpe ]; then
            log "Stage 5: Generate token_list from ${data_feats}/srctexts using BPE"

            # shellcheck disable=SC2002
            cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"

            mkdir -p "${bpedir}"
            # shellcheck disable=SC2002
            <"${data_feats}/srctexts" cut -f 2- -d" "  > "${bpedir}"/train.txt

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
                ${_opts_spm}

            _opts="--bpemodel ${bpemodel}"

        elif [ "${token_type}" = char ]; then
            log "Stage 5: Generate character level token_list from ${data_feats}/srctexts"
            # shellcheck disable=SC2002
            cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
            # TODO: the valid should also be cat together.
            _opts="--non_linguistic_symbols ${nlsyms_txt}"

        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        ${python} -m espnet2.bin.tokenize_text  \
            --token_type "${token_type}" \
            --input "${data_feats}/srctexts" --output "${token_list}" ${_opts} \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"

        # Create word-list for word-LM training
        if ${use_word_lm}; then
            log "Generate word level token_list from ${data_feats}/srctexts"
            ${python} -m espnet2.bin.tokenize_text \
                --token_type word \
                --input "${data_feats}/srctexts" --output "${lm_token_list}" \
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
        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            log "Stage 6: LM collect stats: train_set=${data_feats}/srctexts, dev_set=${lm_dev_text}"
    
            # the srctexts should gets another one to aovid the duplicated keys
            if [ ${spk_num} -ge 2 ]; then
                _spk_list=" "
                for i in $(seq ${spk_num}); do
                    _spk_list+="spk${i} "
                done
                for spk in ${_spk_list}; do
                    awk '{print "'$spk'_"$0}' "${data_feats}/${train_set}/text_${spk}" > "${data_feats}/srctexts_${spk}"
                    awk '{print "'$spk'_"$0}' "${data_feats}/${valid_set}/text_${spk}" > "${data_feats}/valid_srctexts_${spk}"
                    awk '{print "'$spk'_"$0}' "${data_feats}/${test_sets}/text_${spk}" > "${data_feats}/test_srctexts_${spk}"
                done
                cat ${data_feats}/srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/srctexts_with_spk"
                cat ${data_feats}/valid_srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/valid_srctexts_with_spk"
                cat ${data_feats}/test_srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/test_srctexts_with_spk"
                lm_dev_text="${data_feats}/valid_srctexts_with_spk"      # Text file path of language model development set.
                lm_test_text="${data_feats}/test_srctexts_with_spk"    # Text file path of language model evaluation set.
            fi

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
            _nj=$(min "${nj}" "$(<${data_feats}/srctexts_with_spk wc -l)" "$(<${lm_dev_text} wc -l)")

            key_file="${data_feats}/srctexts_with_spk"
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

            # 2. Submit jobs
            log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.

            # shellcheck disable=SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${data_feats}/srctexts_with_spk,text,text" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${lm_args}

            # 3. Aggregate shape files
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


        if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            log "Stage 7: LM Training: train_set=${data_feats}/srctexts, dev_set=${lm_dev_text}"
            if [ ${spk_num} -ge 2 ]; then
                lm_dev_text="${data_feats}/valid_srctexts_with_spk"      # Text file path of language model development set.
                lm_test_text="${data_feats}/test_srctexts_with_spk"    # Text file path of language model evaluation set.
            fi

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
                      --scps "${data_feats}/srctexts_with_spk" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/srctexts_with_spk,text,text "
                _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${data_feats}/srctexts_with_spk,text,text "
                _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
            fi

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

            log "LM training started... log: '${lm_exp}/train.log'"
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${lm_exp}/train.log" \
                --log "${lm_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${joint_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.lm_train \
                    --ngpu "${ngpu}" \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --unused_parameters true \
                    --output_dir "${lm_exp}" \
                    ${_opts} ${lm_args}

        fi


        if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
            log "Stage 8: Calc perplexity: ${lm_test_text}"
            if [ ${spk_num} -ge 2 ]; then
                lm_dev_text="${data_feats}/valid_srctexts_with_spk"      # Text file path of language model development set.
                lm_test_text="${data_feats}/test_srctexts_with_spk"    # Text file path of language model evaluation set.
            fi
            _opts=
            # TODO(kamo): Parallelize?
            log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
            # shellcheck disable=SC2086
            ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m espnet2.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_test_text},text,text" \
                    --train_config "${lm_exp}"/config.yaml \
                    --model_file "${lm_exp}/${decode_lm}" \
                    --output_dir "${lm_exp}/perplexity_test" \
                    ${_opts}
            log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

        fi

    else
        log "Stage 6-8: Skip lm-related stages: use_lm=${use_lm}"
    fi
    # ========================== LM Done here.==========================


    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        _joint_train_dir="${data_feats}/${train_set}"
        _joint_valid_dir="${data_feats}/${valid_set}"
        log "Stage 9: Joint collect stats: train_set=${_joint_train_dir}, valid_set=${_joint_valid_dir}"

        _opts=
        if [ -n "${joint_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${joint_config} "
        fi

        _feats_type="$(<${_joint_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_joint_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${joint_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_joint_train_dir}/${_scp} wc -l)" "$(<${_joint_valid_dir}/${_scp} wc -l)")

        key_file="${_joint_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_joint_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Joint collect-stats started... log: '${_logdir}/stats.*.log'"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_joint_train_dir}/wav.scp,speech_mix,sound "
        _valid_data_param="--valid_data_path_and_name_and_type ${_joint_valid_dir}/wav.scp,speech_mix,sound "
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/utt2category,utt2category,text "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/utt2category,utt2category,text "
        for spk in $(seq "${spk_num}"); do
            if ${use_signal_ref}; then
                _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
            fi
            _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/text_spk${spk},text_ref${spk},text "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/text_spk${spk},text_ref${spk},text "
        done

        if ${use_signal_dereverb_ref}; then
            # reference for dereverberation
            _train_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--train_data_path_and_name_and_type ${_joint_train_dir}/dereverb${n}.scp,dereverb_ref${n},sound "; done)
            _valid_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_joint_valid_dir}/dereverb${n}.scp,dereverb_ref${n},sound "; done)
        fi

        if ${use_signal_noise_ref}; then
            # reference for denoising
            _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--train_data_path_and_name_and_type ${_joint_train_dir}/noise${n}.scp,noise_ref${n},sound "; done)
            _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_joint_valid_dir}/noise${n}.scp,noise_ref${n},sound "; done)
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.


        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.enh_asr_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                ${_train_data_param} \
                ${_valid_data_param} \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${joint_args}

        # 3. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${joint_stats_dir}" --skip_sum_stats

    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _joint_train_dir="${data_feats}/${train_set}"
        _joint_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: Joint model Training: train_set=${_joint_train_dir}, valid_set=${_joint_valid_dir}"

        _opts=
        if [ -n "${joint_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${joint_config} "
        fi

        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((enh_speech_fold_length * 100))"
        # _opts+="--frontend_conf fs=${fs} "

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_joint_train_dir}/wav.scp,speech_mix,sound "
        _train_shape_param="--train_shape_file ${joint_stats_dir}/train/speech_mix_shape "
        _valid_data_param="--valid_data_path_and_name_and_type ${_joint_valid_dir}/wav.scp,speech_mix,sound "
        _valid_shape_param="--valid_shape_file ${joint_stats_dir}/valid/speech_mix_shape "
        _fold_length_param="--fold_length ${_fold_length} "
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/utt2category,utt2category,text "
        _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/utt2category_shape "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/utt2category,utt2category,text "
        _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/utt2category_shape "
        _fold_length_param+="--fold_length ${_fold_length} "
        for spk in $(seq "${spk_num}"); do
            if ${use_signal_ref}; then
                _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
                _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/speech_ref${spk}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            fi
            _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/text_spk${spk},text_ref${spk},text "
            _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/text_ref${spk}_shape "
            if ${use_signal_ref}; then
                _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
                _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/speech_ref${spk}_shape "
            fi
            _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/text_spk${spk},text_ref${spk},text "
            _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/text_ref${spk}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "
        done

        if ${use_signal_dereverb_ref}; then
            # references for dereverberation
            for n in $(seq "${dereverb_ref_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/dereverb${n}.scp,dereverb_ref${n},sound "
                _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/dereverb_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/dereverb${n}.scp,dereverb_ref${n},sound "
                _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/dereverb_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        if ${use_signal_noise_ref}; then
            # references for denoising
            for n in $(seq "${noise_type_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/noise${n}.scp,noise_ref${n},sound "
                _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/noise_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/noise${n}.scp,noise_ref${n},sound "
                _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/noise_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi


        log "enh-asr training started... log: '${joint_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${joint_exp})"
        else
            jobname="${joint_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${joint_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${joint_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.enh_asr_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                ${_train_data_param} \
                ${_valid_data_param} \
                ${_train_shape_param} \
                ${_valid_shape_param} \
                ${_fold_length_param} \
                --resume true \
                --output_dir "${joint_exp}" \
                ${_opts} ${joint_args}

    fi
fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Decoding: training_dir=${joint_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${decode_config}" ]; then
            _opts+="--config ${decode_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${decode_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${decode_lm} "
            fi
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${joint_exp}/${decode_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                _type=sound
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            _decode_data_param="--data_path_and_name_and_type ${_data}/${_scp},speech_mix,${_type} "
            if ${use_signal_ref}; then
                for spk in $(seq "${spk_num}"); do
                    _decode_data_param+="--data_path_and_name_and_type ${_data}/spk${spk}.scp,speech_ref${spk},${_type} "
                done
            fi

            # 1. Split the key file
            key_file=${_data}/'wav.scp'
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/joint_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/joint_inference.JOB.log \
                ${python} -m espnet2.bin.enh_asr_inference \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    ${_decode_data_param} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --joint_train_config "${joint_exp}"/config.yaml \
                    --joint_model_file "${joint_exp}"/"${decode_joint_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${decode_args}

            # 3. Concatenates the output files from each jobs
            for spk in $(seq "${spk_num}"); do
                for f in token token_int score text; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog_spk${spk}/${f}"
                    done | LC_ALL=C sort -k1 >"${_dir}/${f}_spk${spk}"
                done
                if [ -f ${_logdir}/output.1/SDR_spk1 ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/SDR_spk${spk}"
                    done | LC_ALL=C sort -k1 >"${_dir}/SDR_spk${spk}"
                fi
            done

            for protocol in "SDR"; do
                # shellcheck disable=SC2046
                if [ -f "${_dir}"/"${protocol}"_spk1 ]; then
                    paste $(for j in $(seq ${spk_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                    awk 'BEIGN{sum=0}
                        {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                        END{print sum/NR}' > "${_dir}/enh_result_${protocol,,}.txt"
                fi
            done
        done
    fi


    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Scoring"
        if [ "${token_type}" = pnh ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${joint_exp}/decode_${dset}_${decode_tag}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    token_type="word"
                    opts=""
                elif [ "${_type}" = cer ]; then
                    token_type="char"
                    opts=""
                elif [ "${_type}" = ter ]; then
                    token_type="bpe"
                    opts="--bpemodel ${bpemodel} "
                    # opts=${opts}"--cleaner ${cleaner}"
                fi

                for spk in $(seq "${spk_num}"); do
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text_spk${spk}" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type ${token_type} \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ${opts} \
                                ) \
                        <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                            >"${_scoredir}/ref_spk${spk}.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text_spk${spk}"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type ${token_type} \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ${opts} \
                                ) \
                        <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                            >"${_scoredir}/hyp_spk${spk}.trn"
                done

                # PIT scoring
                for r_spk in $(seq "${spk_num}"); do
                    for h_spk in $(seq "${spk_num}"); do
                        sclite \
                            -r "${_scoredir}/ref_spk${r_spk}.trn" trn \
                            -h "${_scoredir}/hyp_spk${h_spk}.trn" trn \
                            -i rm -o all stdout > "${_scoredir}/result_r${r_spk}h${h_spk}.txt"
                    done
                done

                scripts/utils/eval_perm_free_error.py --num-spkrs ${spk_num} \
                    --results-dir ${_scoredir}

                sclite \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        # Show results in Markdown syntax
        scripts/utils/show_asr_result.sh "${joint_exp}" > "${joint_exp}"/RESULTS.md
        cat "${joint_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi

packed_model="${joint_exp}/${joint_exp##*/}_${decode_joint_model%.*}.zip"
if ! "${skip_upload}"; then

    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "[Option] Stage 13: Pack model: ${joint_exp}/packed.zip"

        _opts=

        if "${use_lm}"; then
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${decode_lm} "
                _opts+="--option ${lm_exp}/perplexity_test/ppl "
                _opts+="--option ${lm_exp}/images "
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack asr \
            --asr_train_config "${joint_exp}"/config.yaml \
            --asr_model_file "${joint_exp}"/"${decode_joint_model}" \
            ${_opts} \
            --option "${joint_exp}"/RESULTS.md \
            --option "${joint_exp}"/images \
            --outpath "${packed_model}"

    fi

     if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Upload model to Zenodo: ${packed_model}"

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
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${joint_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${joint_exp}"/RESULTS.md)</code></pre></li>
<li><strong>ASR config</strong><pre><code>$(cat "${joint_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${joint_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi

else
    log "Skip the uploading stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
