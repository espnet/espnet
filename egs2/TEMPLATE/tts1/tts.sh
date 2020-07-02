#!/bin/bash

# Copyright 2019 Tomoki Hayashi
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
stage=1          # Processes starts from the specified stage.
stop_stage=7     # Processes is stopped at the specified stage.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
decode_nj=32     # The number of parallel jobs in decoding.
gpu_decode=false # Whether to perform gpu decoding.
dumpdir=dump     # Directory to dump features.
expdir=exp       # Directory to save experiments.

# Data preparation related
local_data_opts= # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw      # Feature type (fbank or stft or raw).
audio_format=flac   # Audio format (only in feats_type=raw).
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second
# Only used for feats_type != raw
fs=16000          # Sampling rate.
fmin=80           # Minimum frequency of Mel basis.
fmax=7600         # Maximum frequency of Mel basis.
n_mels=80         # The number of mel basis.
n_fft=1024        # The number of fft points.
n_shift=256       # The number of shift points.
win_length=null   # Window length.

oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole

# Training related
train_config= # Config for training.
train_args=   # Arguments for training, e.g., "--max_epoch 1".
              # Note that it will overwrite args in train config.
tag=""        # Suffix for training directory.
num_splits=1  # Number of splitting for tts corpus

# Decoding related
decode_config= # Config for decoding.
decode_args=   # Arguments for decoding, e.g., "--threshold 0.75".
               # Note that it will overwrite args in decode config.
decode_tag=""  # Suffix for decoding directory.
decode_model=train.loss.best.pth # Model path for decoding e.g.,
                                 # decode_model=train.loss.best.pth
                                 # decode_model=3epoch.pth
                                 # decode_model=valid.acc.best.pth
                                 # decode_model=valid.loss.ave.pth
griffin_lim_iters=4 # the number of iterations of Griffin-Lim.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
dev_set=         # Name of development set.
eval_sets=       # Names of evaluation sets. Multiple items can be specified.
srctexts=        # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type.
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
text_fold_length=150   # fold_length for text data
speech_fold_length=800 # fold_length for speech data

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --dev-set "<dev_set_name>" --eval_sets "<eval_set_names>" --srctexts "<srctexts>"

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

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type     # Feature type (fbank or stft or raw, default="${feats_type}").
    --audio_format   # Audio format (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --fs             # Sampling rate (default="${fs}").
    --fmax           # Maximum frequency of Mel basis (default="${fmax}").
    --fmin           # Minimum frequency of Mel basis (default="${fmin}").
    --n_mels         # The number of mel basis (default="${n_mels}").
    --n_fft          # The number of fft points (default="${n_fft}").
    --n_shift        # The number of shift points (default="${n_shift}").
    --win_length     # Window length (default="${win_length}").
    --oov            # Out of vocabrary symbol (default="${oov}").
    --blank          # CTC blank symbol (default="${blank}").
    --sos_eos=       # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config # Config for training (default="${train_config}").
    --train_args   # Arguments for training, e.g., "--max_epoch 1" (default="${train_args}").
                   # Note that it will overwrite args in train config.
    --tag          # Suffix for training directory (default="${tag}").
    --num_splits   # Number of splitting for tts corpus (default="${num_splits}").

    # Decoding related
    --decode_config     # Config for decoding (default="${decode_config}").
    --decode_args       # Arguments for decoding, e.g., "--threshold 0.75" (default="${decode_args}").
                        # Note that it will overwrite args in decode config.
    --decode_tag        # Suffix for decoding directory (default="${decode_tag}").
    --decode_model      # Model path for decoding (default=${decode_model}).
    --griffin_lim_iters # The number of iterations of Griffin-Lim (default=${griffin_lim_iters}).

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set  # Name of training set (required).
    --dev_set    # Name of development set (required).
    --eval_sets  # Names of evaluation sets (required).
                 # Note that multiple items can be specified.
    --srctexts   # Texts to create token list (required).
                 # Note that multiple items can be specified.
    --nlsyms_txt # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type # Transcription type (default="${token_type}").
    --cleaner    # Text cleaner (default="${cleaner}").
    --g2p        # g2p method (default="${g2p}").
    --text_fold_length   # fold_length for text data
    --speech_fold_length # fold_length for speech data
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

# Check feature type
if [ "${feats_type}" = fbank ]; then
    data_feats="${dumpdir}/fbank"
elif [ "${feats_type}" = stft ]; then
    data_feats="${dumpdir}/stft"
elif [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Check token list type
token_listdir="data/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_${token_type}"
    else
        tag="train_${feats_type}_${token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
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
    decode_tag+="_$(echo "${decode_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
tts_stats_dir="${expdir}/tts_stats_${feats_type}_${token_type}"
if [ "${cleaner}" != none ]; then
    tts_stats_dir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    tts_stats_dir+="_${g2p}"
fi
# The directory used for training commands
tts_exp="${expdir}/tts_${tag}"


# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # TODO(kamo): Change kaldi-ark to npy or HDF5?
    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.

    if [ "${feats_type}" = raw ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${dev_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done

    elif [ "${feats_type}" = fbank ] || [ "${feats_type}" = stft ] ; then
        log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}/"

        # Generate the fbank features; by default 80-dimensional fbanks on each frame
        for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${dev_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            # 1. Copy datadir
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"

            # 2. Feature extract
            # TODO(kamo): Wrap (nj->_nj) in make_fbank.sh
            _nj=$(min "${nj}" "$(<${data_feats}${_suf}/${dset}/utt2spk wc -l)")
            _opts=
            if [ "${feats_type}" = fbank ] ; then
                _opts+="--fmax ${fmax} "
                _opts+="--fmin ${fmin} "
                _opts+="--n_mels ${n_mels} "
            fi

            # shellcheck disable=SC2086
            scripts/feats/make_"${feats_type}".sh --cmd "${train_cmd}" --nj "${_nj}" \
                --fs "${fs}" \
                --n_fft "${n_fft}" \
                --n_shift "${n_shift}" \
                --win_length "${win_length}" \
                ${_opts} \
                "${data_feats}${_suf}/${dset}"
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
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

    # NOTE(kamo): Not applying to eval_sets to keep original data
    for dset in "${train_set}" "${dev_set}"; do
        # Copy data dir
        utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
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

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh "${data_feats}/${dset}"
    done

    # shellcheck disable=SC2002
    cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Generate token_list from ${srctexts}"
    # "nlsyms_txt" should be generated by local/data.sh if need

    # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
    # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

    python3 -m espnet2.bin.tokenize_text \
          --token_type "${token_type}" -f 2- \
          --input "${data_feats}/srctexts" --output "${token_list}" \
          --non_linguistic_symbols "${nlsyms_txt}" \
          --cleaner "${cleaner}" \
          --g2p "${g2p}" \
          --write_vocabulary true \
          --add_symbol "${blank}:0" \
          --add_symbol "${oov}:1" \
          --add_symbol "${sos_eos}:-1"
fi

# ========================== Data preparation is done here. ==========================



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    _train_dir="${data_feats}/${train_set}"
    _dev_dir="${data_feats}/${dev_set}"
    log "Stage 5: TTS collect stats: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${train_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
        _opts+="--config ${train_config} "
    fi

    _feats_type="$(<${_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _opts+="--feats_extract fbank "
        _opts+="--feats_extract_conf fs=${fs} "
        _opts+="--feats_extract_conf n_fft=${n_fft} "
        _opts+="--feats_extract_conf fmin=${fmin} "
        _opts+="--feats_extract_conf fmax=${fmax} "
        _opts+="--feats_extract_conf n_mels=${n_mels} "
        _opts+="--feats_extract_conf hop_length=${n_shift} "
        _opts+="--feats_extract_conf win_length=${win_length} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _odim="$(<${_train_dir}/feats_dim)"
        _opts+="--odim=${_odim} "
    fi

    # 1. Split the key file
    _logdir="${tts_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_dev_dir}/${_scp} wc -l)")

    key_file="${_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_dev_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    log "TTS collect_stats started... log: '${_logdir}/stats.*.log'"
    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python3 -m espnet2.bin.tts_train \
            --collect_stats true \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --normalize none \
            --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
            --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_dev_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_dev_dir}/${_scp},speech,${_type}" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${train_args}

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_stats_dir}"

    # Append the num-tokens at the last dimensions. This is used for batch-bins count
    <"${tts_stats_dir}/train/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${tts_stats_dir}/train/text_shape.${token_type}"

    <"${tts_stats_dir}/valid/text_shape" \
        awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
        >"${tts_stats_dir}/valid/text_shape.${token_type}"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    _train_dir="${data_feats}/${train_set}"
    _dev_dir="${data_feats}/${dev_set}"
    log "Stage 6: TTS Training: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${train_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
        _opts+="--config ${train_config} "
    fi

    _feats_type="$(<${_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((speech_fold_length * n_shift))"
        _opts+="--feats_extract fbank "
        _opts+="--feats_extract_conf fs=${fs} "
        _opts+="--feats_extract_conf fmin=${fmin} "
        _opts+="--feats_extract_conf fmax=${fmax} "
        _opts+="--feats_extract_conf n_mels=${n_mels} "
        _opts+="--feats_extract_conf hop_length=${n_shift} "
        _opts+="--feats_extract_conf n_fft=${n_fft} "
        _opts+="--feats_extract_conf win_length=${win_length} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _fold_length="${speech_fold_length}"
        _odim="$(<${_train_dir}/feats_dim)"
        _opts+="--odim=${_odim} "
    fi

    if [ "${num_splits}" -gt 1 ]; then
        # If you met a memory error when parsing text files, this option may help you.
        # The corpus is split into subsets and each subset is used for training one by one in order,
        # so the memory footprint can be limited to the memory required for each dataset.

        _split_dir="${tts_stats_dir}/splits${num_splits}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            python3 -m espnet2.bin.split_scps \
              --scps \
                  "${_train_dir}/text" \
                  "${_train_dir}/${_scp}" \
                  "${tts_stats_dir}/train/speech_shape" \
                  "${tts_stats_dir}/train/text_shape.${token_type}" \
              --num_splits "${num_splits}" \
              --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
        _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
        _opts+="--train_shape_file ${_split_dir}/speech_shape "
        _opts+="--multiple_iterator true "

    else
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
        _opts+="--train_shape_file ${tts_stats_dir}/train/speech_shape "
    fi

    # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

    log "TTS training started... log: '${tts_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${tts_exp}/train.log" \
        --log "${tts_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${tts_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.tts_train \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --normalize global_mvn \
            --normalize_conf "stats_file=${tts_stats_dir}/train/feats_stats.npz" \
            --valid_data_path_and_name_and_type "${_dev_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_dev_dir}/${_scp},speech,${_type}" \
            --valid_shape_file "${tts_stats_dir}/valid/text_shape.${token_type}" \
            --valid_shape_file "${tts_stats_dir}/valid/speech_shape" \
            --resume true \
            --fold_length "${text_fold_length}" \
            --fold_length "${_fold_length}" \
            --output_dir "${tts_exp}" \
            ${_opts} ${train_args}

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Decoding: training_dir=${tts_exp}"

    if ${gpu_decode}; then
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

    _feats_type="$(<${data_feats}/${train_set}/feats_type)"

    # NOTE(kamo): If feats_type=raw, vocoder_conf is unnecessary
    if [ "${_feats_type}" == fbank ] || [ "${_feats_type}" == stft ]; then
        _opts+="--vocoder_conf n_fft=${n_fft} "
        _opts+="--vocoder_conf n_shift=${n_shift} "
        _opts+="--vocoder_conf win_length=${win_length} "
        _opts+="--vocoder_conf fs=${fs} "
    fi
    if [ "${_feats_type}" == fbank ]; then
        _opts+="--vocoder_conf n_mels=${n_mels} "
        _opts+="--vocoder_conf fmin=${fmin} "
        _opts+="--vocoder_conf fmax=${fmax} "
    fi

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${tts_exp}/${decode_tag}_${dset}"
        _logdir="${_dir}/log"
        mkdir -p "${_logdir}"

        # 0. Copy feats_type
        cp "${_data}/feats_type" "${_dir}/feats_type"

        # 1. Split the key file
        key_file=${_data}/text
        split_scps=""
        _nj=$(min "${decode_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/tts_inference.*.log'"
        # shellcheck disable=SC2086
        # NOTE(kan-bayashi): --key_file is useful when we want to use multiple data
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_inference.JOB.log \
            python3 -m espnet2.bin.tts_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/text,text,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --model_file "${tts_exp}"/"${decode_model}" \
                --train_config "${tts_exp}"/config.yaml \
                --output_dir "${_logdir}"/output.JOB \
                --vocoder_conf griffin_lim_iters="${griffin_lim_iters}" \
                ${_opts} ${decode_args}

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"/{norm,denorm,wav,att_ws,probs}
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/norm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/denorm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
        for i in $(seq "${_nj}"); do
            mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
            mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
            mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
            rm -rf "${_logdir}/output.${i}"/{wav,att_ws,probs}
        done
    done

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "[Option] Stage 8: Pack model: ${tts_exp}/packed.tgz"

    python -m espnet2.bin.pack tts \
        --train_config.yaml "${tts_exp}"/config.yaml \
        --model_file.pth "${tts_exp}"/"${decode_model}" \
        --option ${tts_stats_dir}/train/feats_stats.npz  \
        --outpath "${tts_exp}/packed.tgz"

fi


log "Successfully finished. [elapsed=${SECONDS}s]"
