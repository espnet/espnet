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
stage=1          # Processes starts from the specified stage.
stop_stage=12    # Processes is stopped at the specified stage.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
decode_nj=32     # The number of parallel jobs in decoding.
gpu_decode=false # Whether to perform gpu decoding.
dumpdir=dump     # Directory to dump features.
expdir=exp       # Directory to save experiments.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format (only in feats_type=raw).
fs=16k            # Sampling rate.

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
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag=    # Suffix to the result dir for asr model training.
asr_config= # Config for asr model training.
asr_args=   # Arguments for asr model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in asr config.
feats_normalize=global_mvn  # Normalizaton layer type
num_splits_asr=1   # Number of splitting for lm corpus

# Decoding related
decode_tag=    # Suffix to the result dir for decoding.
decode_config= # Config for decoding.
decode_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
               # Note that it will overwrite args in decode config.
decode_lm=valid.loss.best.pth       # Language modle path for decoding.
decode_asr_model=valid.acc.best.pth # ASR model path for decoding.
                                    # e.g.
                                    # decode_asr_model=train.loss.best.pth
                                    # decode_asr_model=3epoch.pth
                                    # decode_asr_model=valid.acc.best.pth
                                    # decode_asr_model=valid.loss.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
train_set=     # Name of training set.
dev_set=       # Name of development set.
eval_sets=     # Names of evaluation sets. Multiple items can be specified.
srctexts=      # Used for the training of BPE and LM and the creation of a vocabulary list.
lm_dev_text=   # Text file path of language model development set.
lm_test_text=  # Text file path of language model evaluation set.
nlsyms_txt=none # Non-linguistic symbol list if existing.
asr_speech_fold_length=800 # fold_length for speech data during ASR training
asr_text_fold_length=150   # fold_length for text data during ASR training
lm_fold_length=150         # fold_length for LM training

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --dev-set <dev_set_name> --eval_sets <eval_set_names> --srctexts <srctexts >

Options:
    # General configuration
    --stage      # Processes starts from the specified stage (default="${stage}").
    --stop_stage # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu       # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes  # The number of nodes
    --nj         # The number of parallel jobs (default="${nj}").
    --decode_nj  # The number of parallel jobs in decoding (default="${decode_nj}").
    --gpu_decode # Whether to perform gpu decoding (default="${gpu_decode}").
    --dumpdir    # Directory to dump features (default="${dumpdir}").
    --expdir     # Directory to save experiments (default="${expdir}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").

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
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training, e.g., "--max_epoch 10" (default="${lm_args}").
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm=1   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ASR model related
    --asr_tag    # Suffix to the result dir for asr model training (default="${asr_tag}").
    --asr_config # Config for asr model training (default="${asr_config}").
    --asr_args   # Arguments for asr model training, e.g., "--max_epoch 10" (default="${asr_args}").
                 # Note that it will overwrite args in asr config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr=1   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --decode_tag       # Suffix to the result dir for decoding (default="${decode_tag}").
    --decode_config    # Config for decoding (default="${decode_config}").
    --decode_args      # Arguments for decoding, e.g., "--lm_weight 0.1" (default="${decode_args}").
                       # Note that it will overwrite args in decode config.
    --decode_lm        # Language modle path for decoding (default="${decode_lm}").
    --decode_asr_model # ASR model path for decoding (default="${decode_asr_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --dev_set       # Name of development set (required).
    --eval_sets     # Names of evaluation sets (required).
    --srctexts      # Used for the training of BPE and LM and the creation of a vocabulary list (required).
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --asr_speech_fold_length # fold_length for speech data during ASR training  (default="${asr_speech_fold_length}").
    --asr_text_fold_length   # fold_length for text data during ASR training  (default="${asr_text_fold_length}").
    --lm_fold_length         # fold_length for LM training  (default="${lm_fold_length}").
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
[ -z "${dev_set}" ] &&   { log "${help_message}"; log "Error: --dev_set is required"  ; exit 2; };
[ -z "${eval_sets}" ] && { log "${help_message}"; log "Error: --eval_sets is required"; exit 2; };
[ -z "${srctexts}" ] &&  { log "${help_message}"; log "Error: --srctexts is required" ; exit 2; };

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

# Use the same text as ASR for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${dev_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${eval_sets%% *}/text"

# Check tokenization type
token_listdir=data/token_list
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/model
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
if [ -z "${asr_tag}" ]; then
    if [ -n "${asr_config}" ]; then
        asr_tag="$(basename "${asr_config}" .yaml)_${feats_type}_${token_type}"
    else
        asr_tag="train_${feats_type}_${token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${asr_args}" ]; then
        asr_tag+="$(echo "${asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
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
    if [ -n "${decode_args}" ]; then
        decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        decode_tag+="_lm_${lm_tag}_$(echo "${decode_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    decode_tag+="_asr_model_$(echo "${decode_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
asr_stats_dir="${expdir}/asr_stats"
lm_stats_dir="${expdir}/lm_stats"
# The directory used for training commands
asr_exp="${expdir}/asr_${asr_tag}"
lm_exp="${expdir}/lm_${lm_tag}"

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
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
        log "Stage 3: Format wav.scp: data/ -> ${data_feats}/org/"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/org/${dset}"
            rm -f ${data_feats}/org/${dset}/{segments,wav.scp,reco2file_and_channel}
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
                "data/${dset}/wav.scp" "${data_feats}/org/${dset}"

            echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank_pitch ]; then
        log "[Require Kaldi] Stage 3: ${feats_type} extract: data/ -> ${data_feats}/org/"

        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            # 1. Copy datadir
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/org/${dset}"

            # 2. Feature extract
            _nj=$(min "${nj}" "$(<"${data_feats}/org/${dset}/utt2spk" wc -l)")
            steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}/org/${dset}"
            utils/fix_data_dir.sh "${data_feats}/org/${dset}"

            # 3. Derive the the frame length and feature dimension
            scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                "${data_feats}/org/${dset}/feats.scp" "${data_feats}/org/${dset}/feats_shape"

            # 4. Write feats_dim
            head -n 1 "${data_feats}/org/${dset}/feats_shape" | awk '{ print $2 }' \
                | cut -d, -f2 > ${data_feats}/org/${dset}/feats_dim

            # 5. Write feats_type
            echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ]; then
        log "Stage 3: ${feats_type} extract: data/ -> ${data_feats}/org/"
        log "${feats_type} is not supported yet."
        exit 1

    elif  [ "${feats_type}" = extracted ]; then
        log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}/"

        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            <data/"${dset}"/cmvn.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"

            pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}/${dset}/feats.scp |" - | \
                awk '{ print $2 }' | cut -d, -f2 > ${data_feats}/${dset}/feats_dim

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
        done

    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do

        # Copy data dir
        utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

        # Remove short utterances
        _feats_type="$(<${data_feats}/${dset}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            min_length=2560

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="$min_length" '{ if ($2 > min_length) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"
        else
            min_length=10

            cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
            <"${data_feats}/org/${dset}/feats_shape" awk -F, ' { print $1 } ' \
                | awk -v min_length="$min_length" '{ if ($2 > min_length) print $0; }' \
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

    # shellcheck disable=SC2002
    cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "${token_type}" = bpe ]; then
        log "Stage 5: Generate token_list from ${data_feats}/srctexts using BPE"

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
        _opts="--non_linguistic_symbols ${nlsyms_txt}"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

    # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
    # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

    python3 -m espnet2.bin.tokenize_text  \
        --token_type "${token_type}" \
        --input "${data_feats}/srctexts" --output "${token_list}" ${_opts} \
        --field 2- \
        --write_vocabulary true \
        --add_symbol "${blank}:0" \
        --add_symbol "${oov}:1" \
        --add_symbol "${sos_eos}:-1"

    # Create word-list for word-LM training
    if ${use_word_lm}; then
        log "Generate word level token_list from ${data_feats}/srctexts"
        python3 -m espnet2.bin.tokenize_text \
            --token_type word \
            --input "${data_feats}/srctexts" --output "${lm_token_list}" \
            --field 2- \
            --write_vocabulary true \
            --vocabulary_size "${word_vocab_size}" \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi

fi


# ========================== Data preparation is done here. ==========================


if "${use_lm}"; then
  if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      log "Stage 6: LM collect stats: train_set=${data_feats}/srctexts, dev_set=${lm_dev_text}"

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
      _nj=$(min "${nj}" "$(<${data_feats}/srctexts wc -l)" "$(<${lm_dev_text} wc -l)")

      key_file="${data_feats}/srctexts"
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
          python3 -m espnet2.bin.lm_train \
              --collect_stats true \
              --use_preprocessor true \
              --bpemodel "${bpemodel}" \
              --token_type "${lm_token_type}"\
              --token_list "${lm_token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --train_data_path_and_name_and_type "${data_feats}/srctexts,text,text" \
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
      python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

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
              python3 -m espnet2.bin.split_scps \
                --scps "${data_feats}/srctexts" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                --num_splits "${num_splits_lm}" \
                --output_dir "${_split_dir}"
              touch "${_split_dir}/.done"
          else
              log "${_split_dir}/.done exists. Spliting is skipped"
          fi

          _opts+="--train_data_path_and_name_and_type ${_split_dir}/srctexts,text,text "
          _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
          _opts+="--multiple_iterator true "

      else
          _opts+="--train_data_path_and_name_and_type ${data_feats}/srctexts,text,text "
          _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
      fi

      # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

      log "LM training started... log: '${lm_exp}/train.log'"
      # shellcheck disable=SC2086
      python3 -m espnet2.bin.launch \
          --cmd "${cuda_cmd} --name ${lm_exp}/train.log" \
          --log "${lm_exp}"/train.log \
          --ngpu "${ngpu}" \
          --num_nodes "${num_nodes}" \
          --init_file_prefix "${asr_exp}"/.dist_init_ \
          --multiprocessing_distributed true -- \
          python3 -m espnet2.bin.lm_train \
              --ngpu "${ngpu}" \
              --use_preprocessor true \
              --bpemodel "${bpemodel}" \
              --token_type "${lm_token_type}"\
              --token_list "${lm_token_list}" \
              --non_linguistic_symbols "${nlsyms_txt}" \
              --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
              --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
              --fold_length "${lm_fold_length}" \
              --resume true \
              --output_dir "${lm_exp}" \
              ${_opts} ${lm_args}

  fi


  if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
      log "Stage 8: Calc perplexity: ${lm_test_text}"
      _opts=
      # TODO(kamo): Parallelize?
      log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
      # shellcheck disable=SC2086
      ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
          python3 -m espnet2.bin.lm_calc_perplexity \
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


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    _asr_train_dir="${data_feats}/${train_set}"
    _asr_dev_dir="${data_feats}/${dev_set}"
    log "Stage 9: ASR collect stats: train_set=${_asr_train_dir}, dev_set=${_asr_dev_dir}"

    _opts=
    if [ -n "${asr_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
        _opts+="--config ${asr_config} "
    fi

    _feats_type="$(<${_asr_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
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
    _nj=$(min "${nj}" "$(<${_asr_train_dir}/${_scp} wc -l)" "$(<${_asr_dev_dir}/${_scp} wc -l)")

    key_file="${_asr_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_asr_dev_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.

    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python3 -m espnet2.bin.asr_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/${_scp},speech,${_type}" \
            --train_data_path_and_name_and_type "${_asr_train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/text,text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${asr_args}

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${asr_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    <"${asr_stats_dir}/train/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${asr_stats_dir}/train/text_shape.${token_type}"

    <"${asr_stats_dir}/valid/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${asr_stats_dir}/valid/text_shape.${token_type}"
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    _asr_train_dir="${data_feats}/${train_set}"
    _asr_dev_dir="${data_feats}/${dev_set}"
    log "Stage 10: ASR Training: train_set=${_asr_train_dir}, dev_set=${_asr_dev_dir}"

    _opts=
    if [ -n "${asr_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
        _opts+="--config ${asr_config} "
    fi

    _feats_type="$(<${_asr_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((asr_speech_fold_length * 100))"
        _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _fold_length="${asr_speech_fold_length}"
        _input_size="$(<${_asr_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        # Default normalization is utterance_mvn and changes to global_mvn
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
    fi

    if [ "${num_splits_asr}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            python3 -m espnet2.bin.split_scps \
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

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

    log "ASR training started... log: '${asr_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${asr_exp}/train.log" \
        --log "${asr_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${asr_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.asr_train \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_asr_dev_dir}/text,text,text" \
            --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
            --resume true \
            --fold_length "${_fold_length}" \
            --fold_length "${asr_text_fold_length}" \
            --output_dir "${asr_exp}" \
            ${_opts} ${asr_args}

fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    log "Stage 11: Decoding: training_dir=${asr_exp}"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
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

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/decode_${dset}_${decode_tag}"
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

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""
        _nj=$(min "${decode_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
        # shellcheck disable=SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
            python3 -m espnet2.bin.asr_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --asr_train_config "${asr_exp}"/config.yaml \
                --asr_model_file "${asr_exp}"/"${decode_asr_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${decode_args}

        # 3. Concatenates the output files from each jobs
        for f in token token_int score text; do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}"
            done | LC_ALL=C sort -k1 >"${_dir}/${f}"
        done
    done
fi


if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: Scoring"

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${asr_exp}/decode_${dset}_${decode_tag}"

        for _type in cer wer ter; do
            [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            if [ "${_type}" = wer ]; then
                # Tokenize text to word level
                paste \
                    <(<"${_data}/text" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true) \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                paste \
                    <(<"${_dir}/text"  \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type word \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true) \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"


            elif [ "${_type}" = cer ]; then
                # Tokenize text to char level
                paste \
                    <(<"${_data}/text" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true) \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                paste \
                    <(<"${_dir}/text"  \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type char \
                              --non_linguistic_symbols "${nlsyms_txt}" \
                              --remove_non_linguistic_symbols true) \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            elif [ "${_type}" = ter ]; then
                # Tokenize text using BPE
                paste \
                    <(<"${_data}/text" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type bpe \
                              --bpemodel "${bpemodel}") \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                paste \
                    <(<"${_dir}/text" \
                          python3 -m espnet2.bin.tokenize_text  \
                              -f 2- --input - --output - \
                              --token_type bpe \
                              --bpemodel "${bpemodel}") \
                    <(<"${_data}/text" awk '{ print "(" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"
            fi

            sclite \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done

    done

    # Show results in Markdown syntax
    scripts/utils/show_asr_result.sh "${asr_exp}" > "${asr_exp}"/RESULTS.md
    cat "${asr_exp}"/RESULTS.md

fi


if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "[Option] Stage 13: Pack model: ${asr_exp}/packed.tgz"

    _opts=
    if "${use_lm}"; then
        _opts+="--lm_train_config.yaml ${lm_exp}/config.yaml "
        _opts+="--lm_file.pth ${lm_exp}/${decode_lm} "
    fi
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${asr_stats_dir}/train/feats_stats.npz "
    fi
    if [ "${token_type}" = bpe ]; then
        _opts+="--option ${bpemodel} "
    fi
    # shellcheck disable=SC2086
    python -m espnet2.bin.pack asr \
        --asr_train_config.yaml "${asr_exp}"/config.yaml \
        --asr_model_file.pth "${asr_exp}"/"${decode_asr_model}" \
        ${_opts} \
        --option "${asr_exp}"/RESULTS.md \
        --outpath "${asr_exp}/packed.tgz"

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
