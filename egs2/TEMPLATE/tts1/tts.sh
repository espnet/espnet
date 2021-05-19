#!/usr/bin/env bash

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
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (fbank or stft or raw).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.
use_xvector=false    # Whether to use x-vector (Require Kaldi).
# Only used for feats_type != raw
fs=16000          # Sampling rate.
fmin=80           # Minimum frequency of Mel basis.
fmax=7600         # Maximum frequency of Mel basis.
n_mels=80         # The number of mel basis.
n_fft=1024        # The number of fft points.
n_shift=256       # The number of shift points.
win_length=null   # Window length.
# Only used for the model using pitch features (e.g. FastSpeech2)
f0min=80          # Maximum f0 for pitch extraction.
f0max=400         # Minimum f0 for pitch extraction.

oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
tts_exp=""         # Specify the direcotry path for experiment. If this option is specified, tag is ignored.
tts_stats_dir=""   # Specify the direcotry path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for tts corpus.
teacher_dumpdir="" # Directory of teacher outputs (needed if tts=fastspeech).
write_collected_feats=false # Whether to dump features in stats collection.

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding, e.g., "--threshold 0.75".
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
griffin_lim_iters=4 # the number of iterations of Griffin-Lim.
download_model=""   # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""      # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type.
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
text_fold_length=150   # fold_length for text data.
speech_fold_length=800 # fold_length for speech data.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

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

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (fbank or stft or raw, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --use_xvector      # Whether to use X-vector (Require Kaldi, default="${use_xvector}").
    --fs               # Sampling rate (default="${fs}").
    --fmax             # Maximum frequency of Mel basis (default="${fmax}").
    --fmin             # Minimum frequency of Mel basis (default="${fmin}").
    --n_mels           # The number of mel basis (default="${n_mels}").
    --n_fft            # The number of fft points (default="${n_fft}").
    --n_shift          # The number of shift points (default="${n_shift}").
    --win_length       # Window length (default="${win_length}").
    --f0min            # Maximum f0 for pitch extraction (default="${f0min}").
    --f0max            # Minimum f0 for pitch extraction (default="${f0max}").
    --oov              # Out of vocabrary symbol (default="${oov}").
    --blank            # CTC blank symbol (default="${blank}").
    --sos_eos          # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --tts_exp       # Specify the direcotry path for experiment.
                    # If this option is specified, tag is ignored (default="${tts_exp}").
    --tts_stats_dir # Specify the direcotry path for statistics.
                    # If empty, automatically decided (default="${tts_stats_dir}").
    --num_splits    # Number of splitting for tts corpus (default="${num_splits}").
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --griffin_lim_iters # The number of iterations of Griffin-Lim (default=${griffin_lim_iters}).
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --srctexts           # Texts to create token list (required).
                         # Note that multiple items can be specified.
    --nlsyms_txt         # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type         # Transcription type (default="${token_type}").
    --cleaner            # Text cleaner (default="${cleaner}").
    --g2p                # g2p method (default="${g2p}").
    --lang               # The language type of corpus (default="${lang}").
    --text_fold_length   # Fold length for text data (default="${text_fold_length}").
    --speech_fold_length # Fold length for speech data (default="${speech_fold_length}").
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
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
if [ -z "${tts_stats_dir}" ]; then
    tts_stats_dir="${expdir}/tts_stats_${feats_type}_${token_type}"
    if [ "${cleaner}" != none ]; then
        tts_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tts_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${tts_exp}" ]; then
    tts_exp="${expdir}/tts_${tag}"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
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
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
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
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
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

        # Extract X-vector
        if "${use_xvector}"; then
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
                # 1. Copy datadir and resample to 16k
                utils/copy_data_dir.sh "data/${dset}" "${dumpdir}/mfcc/${dset}"
                utils/data/resample_data_dir.sh 16000 "${dumpdir}/mfcc/${dset}"

                # 2. Extract mfcc features
                _nj=$(min "${nj}" "$(<${dumpdir}/mfcc/${dset}/utt2spk wc -l)")
                steps/make_mfcc.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    --write-utt2num-frames true \
                    --mfcc-config conf/mfcc.conf \
                    "${dumpdir}/mfcc/${dset}"
                utils/fix_data_dir.sh "${dumpdir}/mfcc/${dset}"

                # 3. Compute VAD decision
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
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                cp "${data_feats}${_suf}/${dset}"/wav.{scp,scp.bak}
                <"${data_feats}${_suf}/${dset}/wav.scp.bak" \
                    utils/filter_scp.pl "${dumpdir}/xvector/${dset}/xvector.scp" \
                    >"${data_feats}${_suf}/${dset}/wav.scp"
                utils/fix_data_dir.sh "${data_feats}${_suf}/${dset}"
            done
        fi
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
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

            # Filter x-vector
            if "${use_xvector}"; then
                cp "${dumpdir}/xvector/${dset}"/xvector.{scp,scp.bak}
                <"${dumpdir}/xvector/${dset}/xvector.scp.bak" \
                    utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
                    >"${dumpdir}/xvector/${dset}/xvector.scp"
            fi
        done

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        ${python} -m espnet2.bin.tokenize_text \
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
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: TTS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _feats_type="$(<${_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
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

        # Add extra configs for additional inputs
        # NOTE(kan-bayashi): We always pass this options but not used in default
        _opts+="--pitch_extract_conf fs=${fs} "
        _opts+="--pitch_extract_conf n_fft=${n_fft} "
        _opts+="--pitch_extract_conf hop_length=${n_shift} "
        _opts+="--pitch_extract_conf f0max=${f0max} "
        _opts+="--pitch_extract_conf f0min=${f0min} "
        _opts+="--energy_extract_conf fs=${fs} "
        _opts+="--energy_extract_conf n_fft=${n_fft} "
        _opts+="--energy_extract_conf hop_length=${n_shift} "
        _opts+="--energy_extract_conf win_length=${win_length} "

        if [ -n "${teacher_dumpdir}" ]; then
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
        fi

        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/xvector/${train_set}"
            _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        # 1. Split the key file
        _logdir="${tts_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_valid_dir}/${_scp} wc -l)")

        key_file="${_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${tts_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${tts_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${tts_stats_dir}/run.sh"; chmod +x "${tts_stats_dir}/run.sh"

        # 3. Submit jobs
        log "TTS collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.tts_train \
                --collect_stats true \
                --write_collected_feats "${write_collected_feats}" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize none \
                --pitch_normalize none \
                --energy_normalize none \
                --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},speech,${_type}" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${train_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_stats_dir}"

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
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: TTS Training: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if [ -z "${teacher_dumpdir}" ]; then
            #####################################
            #     CASE 1: AR model training     #
            #####################################
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
                    ${python} -m espnet2.bin.split_scps \
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
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/speech_shape "
        else
            #####################################
            #   CASE 2: Non-AR model training   #
            #####################################
            _teacher_train_dir="${teacher_dumpdir}/${train_set}"
            _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
            _fold_length="${speech_fold_length}"

            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
            _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
            _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
            _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
            _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "

            if [ -e ${_teacher_train_dir}/probs ]; then
                # Knowledge distillation case: use the outputs of the teacher model as the target
                _scp=feats.scp
                _type=npy
                _odim="$(head -n 1 "${_teacher_train_dir}/speech_shape" | cut -f 2 -d ",")"
                _opts+="--odim=${_odim} "
                _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/denorm/${_scp},speech,${_type} "
                _opts+="--train_shape_file ${_teacher_train_dir}/speech_shape "
                _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/denorm/${_scp},speech,${_type} "
                _opts+="--valid_shape_file ${_teacher_valid_dir}/speech_shape "
            else
                # Teacher forcing case: use groundtruth as the target
                if [ "${feats_type}" = raw ]; then
                    _scp=wav.scp
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
                    _odim="$(head -n 1 "${tts_stats_dir}/train/speech_shape" | cut -f 2 -d ",")"
                    _opts+="--odim=${_odim} "
                fi
                _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
                _opts+="--train_shape_file ${tts_stats_dir}/train/speech_shape "
                _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
                _opts+="--valid_shape_file ${tts_stats_dir}/valid/speech_shape "
            fi
        fi

        # If there are dumped files of additional inputs, we use it to reduce computational cost
        # NOTE (kan-bayashi): Use dumped files of the target features as well?
        if [ -e "${tts_stats_dir}/train/collect_feats/pitch.scp" ]; then
            _scp=pitch.scp
            _type=npy
            _train_collect_dir=${tts_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},pitch,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},pitch,${_type} "
        fi
        if [ -e "${tts_stats_dir}/train/collect_feats/energy.scp" ]; then
            _scp=energy.scp
            _type=npy
            _train_collect_dir=${tts_stats_dir}/train/collect_feats
            _valid_collect_dir=${tts_stats_dir}/valid/collect_feats
            _opts+="--train_data_path_and_name_and_type ${_train_collect_dir}/${_scp},energy,${_type} "
            _opts+="--valid_data_path_and_name_and_type ${_valid_collect_dir}/${_scp},energy,${_type} "
        fi

        # Check extra statistics
        if [ -e "${tts_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+="--pitch_extract_conf fs=${fs} "
            _opts+="--pitch_extract_conf n_fft=${n_fft} "
            _opts+="--pitch_extract_conf hop_length=${n_shift} "
            _opts+="--pitch_extract_conf f0max=${f0max} "
            _opts+="--pitch_extract_conf f0min=${f0min} "
            _opts+="--pitch_normalize_conf stats_file=${tts_stats_dir}/train/pitch_stats.npz "
        fi
        if [ -e "${tts_stats_dir}/train/energy_stats.npz" ]; then
            _opts+="--energy_extract_conf fs=${fs} "
            _opts+="--energy_extract_conf n_fft=${n_fft} "
            _opts+="--energy_extract_conf hop_length=${n_shift} "
            _opts+="--energy_extract_conf win_length=${win_length} "
            _opts+="--energy_normalize_conf stats_file=${tts_stats_dir}/train/energy_stats.npz "
        fi

        # Add X-vector to the inputs if needed
        if "${use_xvector}"; then
            _xvector_train_dir="${dumpdir}/xvector/${train_set}"
            _xvector_valid_dir="${dumpdir}/xvector/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        log "Generate '${tts_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${tts_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${tts_exp}/run.sh"; chmod +x "${tts_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "TTS training started... log: '${tts_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${tts_exp})"
        else
            jobname="${tts_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${tts_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tts_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.tts_train \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize global_mvn \
                --normalize_conf "stats_file=${tts_stats_dir}/train/feats_stats.npz" \
                --resume true \
                --fold_length "${text_fold_length}" \
                --fold_length "${_fold_length}" \
                --output_dir "${tts_exp}" \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    tts_exp="${expdir}/${download_model}"
    mkdir -p "${tts_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${tts_exp}/config.txt"

    # Get the path of each file
    _model_file=$(<"${tts_exp}/config.txt" sed -e "s/.*'model_file': '\([^']*\)'.*$/\1/")
    _train_config=$(<"${tts_exp}/config.txt" sed -e "s/.*'train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_model_file}" "${tts_exp}"
    ln -sf "${_train_config}" "${tts_exp}"
    inference_model=$(basename "${_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Decoding: training_dir=${tts_exp}"

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

        if [ -z "${teacher_dumpdir}" ]; then
            _feats_type="$(<${data_feats}/${train_set}/feats_type)"
        else
            if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
                # Knowledge distillation
                _feats_type=fbank
            else
                # Teacher forcing
                _feats_type="$(<${data_feats}/${train_set}/feats_type)"
            fi
        fi

        # NOTE(kamo): If feats_type=raw, vocoder_conf is unnecessary
        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        if [ "${_feats_type}" = fbank ] || [ "${_feats_type}" = stft ]; then
            _opts+="--vocoder_conf n_fft=${n_fft} "
            _opts+="--vocoder_conf n_shift=${n_shift} "
            _opts+="--vocoder_conf win_length=${win_length} "
            _opts+="--vocoder_conf fs=${fs} "
            _scp=feats.scp
            _type=kaldi_ark
        fi
        if [ "${_feats_type}" = fbank ]; then
            _opts+="--vocoder_conf n_mels=${n_mels} "
            _opts+="--vocoder_conf fmin=${fmin} "
            _opts+="--vocoder_conf fmax=${fmax} "
        fi

        log "Generate '${tts_exp}/${inference_tag}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${tts_exp}/${inference_tag}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${tts_exp}/${inference_tag}/run.sh"; chmod +x "${tts_exp}/${inference_tag}/run.sh"


        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _speech_data="${_data}"
            _dir="${tts_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/log"
            mkdir -p "${_logdir}"

            _ex_opts=""
            if [ -n "${teacher_dumpdir}" ]; then
                # Use groundtruth of durations
                _teacher_dir="${teacher_dumpdir}/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_teacher_dir}/durations,durations,text_int "
                # Overwrite speech arguments if use knowledge distillation
                if [ -e "${teacher_dumpdir}/${train_set}/probs" ]; then
                    _speech_data="${_teacher_dir}/denorm"
                    _scp=feats.scp
                    _type=npy
                fi
            fi

            # Add X-vector to the inputs if needed
            if "${use_xvector}"; then
                _xvector_dir="${dumpdir}/xvector/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_xvector_dir}/xvector.scp,spembs,kaldi_ark "
            fi

            # 0. Copy feats_type
            cp "${_data}/feats_type" "${_dir}/feats_type"

            # 1. Split the key file
            key_file=${_data}/text
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 3. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/tts_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_inference.JOB.log \
                ${python} -m espnet2.bin.tts_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/text,text,text" \
                    --data_path_and_name_and_type ${_speech_data}/${_scp},speech,${_type} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --model_file "${tts_exp}"/"${inference_model}" \
                    --train_config "${tts_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    --vocoder_conf griffin_lim_iters="${griffin_lim_iters}" \
                    ${_opts} ${_ex_opts} ${inference_args}

            # 4. Concatenates the output files from each jobs
            mkdir -p "${_dir}"/{norm,denorm,wav}
            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/norm/feats.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/denorm/feats.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
            for i in $(seq "${_nj}"); do
                 cat "${_logdir}/output.${i}/speech_shape/speech_shape"
            done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                rm -rf "${_logdir}/output.${i}"/wav
            done
            if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
                mkdir -p "${_dir}"/att_ws
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/durations/durations"
                done | LC_ALL=C sort -k1 > "${_dir}/durations"
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/focus_rates/focus_rates"
                done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                    rm -rf "${_logdir}/output.${i}"/att_ws
                done
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
else
    log "Skip the evaluation stages"
fi


packed_model="${tts_exp}/${tts_exp##*/}_${inference_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Pack model: ${packed_model}"

        _opts=""
        if [ -e "${tts_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+=" --option ${tts_stats_dir}/train/pitch_stats.npz"
        fi
        if [ -e "${tts_stats_dir}/train/energy_stats.npz" ]; then
            _opts+=" --option ${tts_stats_dir}/train/energy_stats.npz"
        fi
        if "${use_xvector}"; then
            for dset in "${train_set}" ${test_sets}; do
                _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.scp"
                _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.ark"
            done
        fi
        ${python} -m espnet2.bin.pack tts \
            --train_config "${tts_exp}"/config.yaml \
            --model_file "${tts_exp}"/"${inference_model}" \
            --option "${tts_stats_dir}"/train/feats_stats.npz  \
            --option "${tts_exp}"/images  \
            --outpath "${packed_model}" \
            ${_opts}

        # NOTE(kamo): If you'll use packed model to inference in this script, do as follows
        #   % unzip ${packed_model}
        #   % ./run.sh --stage 8 --tts_exp $(basename ${packed_model} .zip) --inference_model pretrain.pth
    fi


    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Signup to Zenodo: https://zenodo.org/
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
        # /some/where/espnet/egs2/foo/tts1/ -> foo/tt1
        _task="$(pwd | rev | cut -d/ -f1-2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${tts_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
# Download the model file here
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Config</strong><pre><code>$(cat "${tts_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${tts_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
