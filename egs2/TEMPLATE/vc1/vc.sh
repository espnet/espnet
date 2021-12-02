#!/usr/bin/env bash

# Copyright 2021 Peter Wu
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
skip_upload_hf=true  # Skip uploading to huggingface stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
# TODO
backend=pytorch
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=raw             # Input feature type.
audio_format=flac          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=20        # Maximum duration in second.
use_xvector=false          # Whether to use x-vector (Require Kaldi).
use_sid=false              # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false              # Whether to use language id as the inputs (Need utt2lang in data directory).
feats_extract=fbank        # On-the-fly feature extractor.
feats_normalize=global_mvn # On-the-fly feature normalizer.
fs=16000                   # Sampling rate.
n_fft=1024                 # The number of fft points.
n_shift=256                # The number of shift points.
win_length=null            # Window length.
fmin=80                    # Minimum frequency of Mel basis.
fmax=7600                  # Maximum frequency of Mel basis.
n_mels=80                  # The number of mel basis.
# Only used for the model using pitch & energy features (e.g. FastSpeech2)
f0min=80  # Maximum f0 for pitch extraction.
f0max=400 # Minimum f0 for pitch extraction.

# Vocabulary related
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Training related
train_config=""    # Config for training.
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
vc_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
vc_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for vc corpus.
teacher_dumpdir="" # Directory of teacher outputs (needed if vc=fastspeech).
write_collected_feats=false # Whether to dump features in stats collection.
vc_task=vc                # VC task (vc or gan_vc).

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model=""  # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
src_train_set=""     # Name of src training set.
src_valid_set=""     # Name of src validation set used for monitoring/tuning network training.
src_test_sets=""     # Names of src test sets. Multiple items (e.g., both dev and eval sets) can be specified.
trg_train_set=""     # Name of src training set.
trg_valid_set=""     # Name of src validation set used for monitoring/tuning network training.
trg_test_sets=""     # Names of src test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""      # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type (char or phn).
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
text_fold_length=150   # fold_length for text data.
speech_fold_length=800 # fold_length for speech data.

# TODO
# decoding related
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage

# TODO
# pretrained model related
pretrained_model=           # available pretrained models: m_ailabs.judy.vtn_tts_pt

 # TODO
# dataset configuration
srcspk=clb                  # available speakers: "slt" "clb" "bdl" "rms"
trgspk=slt

# TODO
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
    --feats_type       # Feature type (default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --use_xvector      # Whether to use X-vector (Require Kaldi, default="${use_xvector}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --feats_extract    # On the fly feature extractor (default="${feats_extract}").
    --feats_normalize  # Feature normalizer for on the fly feature extractor (default="${feats_normalize}")
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
    --vc_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${vc_exp}").
    --vc_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${vc_stats_dir}").
    --num_splits    # Number of splitting for vc corpus (default="${num_splits}").
    --teacher_dumpdir       # Directory of teacher outputs (needed if vc=fastspeech, default="${teacher_dumpdir}").
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").
    --vc_task              # VC task {vc or gan_vc} (default="${vc_task}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --vocoder_file      # Vocoder paramemter file (default=${vocoder_file}).
                        # If set to none, Griffin-Lim vocoder will be used.
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --src_train_set          # Name of training set (required).
    --src_valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --src_test_sets          # Names of test sets (required).
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
if [ "${feats_type}" = raw ]; then
    data_feats="${dumpdir}/raw"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# Check token list type
token_listdir="${dumpdir}/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"

# Check old version token list dir existence
if [ -e data/token_list ] && [ ! -e "${dumpdir}/token_list" ]; then
    log "Default token_list directory path is changed from data to ${dumpdir}."
    log "Copy data/token_list to ${dumpdir}/token_list for the compatibility."
    [ ! -e ${dumpdir} ] && mkdir -p ${dumpdir}
    cp -a "data/token_list" "${dumpdir}/token_list"
fi

# TODO
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
if [ -z "${vc_stats_dir}" ]; then
    vc_stats_dir="${expdir}/vc_stats_${feats_type}"
    if [ "${feats_extract}" != fbank ]; then
        vc_stats_dir+="_${feats_extract}"
    fi
    vc_stats_dir+="_${token_type}"
    if [ "${cleaner}" != none ]; then
        vc_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        vc_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${vc_exp}" ]; then
    vc_exp="${expdir}/vc_${tag}"
fi


# ========================== Main stages start from here. ==========================

pair=${srcspk}_${trgspk}
pair_train_set=${pair}_train
pair_valid_set=${pair}_dev
pair_test_set=${pair}_eval

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/*"
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${src_train_set}" "${src_valid_set}" ${src_test_sets} "${trg_train_set}" "${trg_valid_set}" ${trg_test_sets}; do
            if [ "${dset}" = "${src_train_set}" ] || [ "${dset}" = "${src_valid_set}" ] || [ "${dset}" = "${trg_train_set}" ] || [ "${dset}" = "${trg_valid_set}" ]; then
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

        # NOTE(peter): skipped Extract X-vector

        # NOTE(peter): skipped Prepare spk id input

        # NOTE(peter): skipped Prepare lang id input
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${src_train_set}" "${src_valid_set}" "${trg_train_set}" "${trg_valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                cp "${data_feats}/org/${dset}/utt2sid" "${data_feats}/${dset}/utt2sid"
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                cp "${data_feats}/org/${dset}/utt2lid" "${data_feats}/${dset}/utt2lid"
            fi

            # Remove short utterances
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

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            _fix_opts=""
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                _fix_opts="--utt_extra_files utt2sid "
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                _fix_opts="--utt_extra_files utt2lid "
            fi
            # shellcheck disable=SC2086
            utils/fix_data_dir.sh ${_fix_opts} "${data_feats}/${dset}"

            # Filter x-vector
            if "${use_xvector}"; then
                cp "${dumpdir}/xvector/${dset}"/xvector.{scp,scp.bak}
                <"${dumpdir}/xvector/${dset}/xvector.scp.bak" \
                    utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
                    >"${dumpdir}/xvector/${dset}/xvector.scp"
            fi
        done
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"

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
        log "Stage 5: VC collect stats"

        _train_sets=( "${src_train_set}" "${trg_train_set}" )
        _valid_sets=( "${src_valid_set}" "${trg_valid_set}" )
        _set_ids=( "src" "trg" )

        for i in "${!_train_sets[@]}"; do
            _train_dir="${data_feats}/${_train_sets[i]}"
            _valid_dir="${data_feats}/${_valid_sets[i]}"
            _set_id="${_set_ids[i]}"

            _opts=
            if [ -n "${train_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
                _opts+="--config ${train_config} "
            fi

            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--feats_extract ${feats_extract} "
            _opts+="--feats_extract_conf n_fft=${n_fft} "
            _opts+="--feats_extract_conf hop_length=${n_shift} "
            _opts+="--feats_extract_conf win_length=${win_length} "
            if [ "${feats_extract}" = fbank ]; then
                _opts+="--feats_extract_conf fs=${fs} "
                _opts+="--feats_extract_conf fmin=${fmin} "
                _opts+="--feats_extract_conf fmax=${fmax} "
                _opts+="--feats_extract_conf n_mels=${n_mels} "
            fi
            _opts+="--input_feats_extract ${feats_extract} "
            _opts+="--input_feats_extract_conf n_fft=${n_fft} "
            _opts+="--input_feats_extract_conf hop_length=${n_shift} "
            _opts+="--input_feats_extract_conf win_length=${win_length} "
            if [ "${feats_extract}" = fbank ]; then
                _opts+="--input_feats_extract_conf fs=${fs} "
                _opts+="--input_feats_extract_conf fmin=${fmin} "
                _opts+="--input_feats_extract_conf fmax=${fmax} "
                _opts+="--input_feats_extract_conf n_mels=${n_mels} "
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
                _teacher_train_dir="${teacher_dumpdir}/${_train_sets[i]}"
                _teacher_valid_dir="${teacher_dumpdir}/${_valid_sets[i]}"
                _opts+="--train_data_path_and_name_and_type ${_teacher_train_dir}/durations,durations,text_int "
                _opts+="--valid_data_path_and_name_and_type ${_teacher_valid_dir}/durations,durations,text_int "
            fi

            if "${use_xvector}"; then
                _xvector_train_dir="${dumpdir}/xvector/${_train_sets[i]}"
                _xvector_valid_dir="${dumpdir}/xvector/${_valid_sets[i]}"
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

            # 1. Split the key file
            _logdir="${vc_stats_dir}/logdir"
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
            log "Generate '${vc_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
            mkdir -p "${vc_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${vc_stats_dir}/run.sh"; chmod +x "${vc_stats_dir}/run.sh"

            # 3. Submit jobs
            log "VC collect_stats started... log: '${_logdir}/stats.*.log'"
            # shellcheck disable=SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m "espnet2.bin.tts_train" \
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
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${vc_stats_dir}"

            mv "${vc_stats_dir}/train" "${vc_stats_dir}/${_set_id}_train"
            mv "${vc_stats_dir}/valid" "${vc_stats_dir}/${_set_id}_valid"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${vc_stats_dir}/${_set_id}_train/text_shape" \
                awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
                >"${vc_stats_dir}/${_set_id}_train/text_shape.${token_type}"

            <"${vc_stats_dir}/${_set_id}_valid/text_shape" \
                awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
                >"${vc_stats_dir}/${_set_id}_valid/text_shape.${token_type}"
        done
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: VC Training"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((speech_fold_length * n_shift))"
        _opts+="--feats_extract ${feats_extract} "
        _opts+="--feats_extract_conf n_fft=${n_fft} "
        _opts+="--feats_extract_conf hop_length=${n_shift} "
        _opts+="--feats_extract_conf win_length=${win_length} "
        if [ "${feats_extract}" = fbank ]; then
            _opts+="--feats_extract_conf fs=${fs} "
            _opts+="--feats_extract_conf fmin=${fmin} "
            _opts+="--feats_extract_conf fmax=${fmax} "
            _opts+="--feats_extract_conf n_mels=${n_mels} "
        fi

        # TODO
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
        _opts+="--train_shape_file ${tts_stats_dir}/train/speech_shape "
    
        # TODO
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/${_scp},speech,${_type} "
        _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "
        _opts+="--valid_shape_file ${tts_stats_dir}/valid/speech_shape "
        
        # TODO this and below
        # If there are dumped files of additional inputs, we use it to reduce computational cost
        # NOTE (kan-bayashi): Use dumped files of the target features as well?

        log "Generate '${vc_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${vc_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${vc_exp}/run.sh"; chmod +x "${vc_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "VC training started... log: '${vc_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${vc_exp})"
        else
            jobname="${vc_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${vc_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${vc_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.${vc_task}_train" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize "${feats_normalize}" \
                --resume true \
                --fold_length "${text_fold_length}" \
                --fold_length "${_fold_length}" \
                --output_dir "${vc_exp}" \
                ${_opts} ${train_args}

    fi
fi

# [WIP]

if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

        if [ -z "${inference_model}" ]; then
            inference_model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
            inference_model=$(basename ${inference_model})
        fi
        outdir=${expdir}/outputs_${inference_model}_$(basename ${inference_config%.*})
        if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
            echo "stage 5: Decoding and synthesis"

            echo "Decoding..."
            pids=() # initialize pids
            for name in ${pair_valid_set} ${pair_test_set}; do
            (
                [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
                cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}
                splitjson.py --parts ${nj} ${outdir}/${name}/data.json
                # decode in parallel
                ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
                    vc_decode.py \
                        --backend ${backend} \
                        --ngpu 0 \
                        --verbose ${verbose} \
                        --out ${outdir}/${name}/feats.JOB \
                        --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                        --model ${expdir}/results/${inference_model} \
                        --config ${inference_config}
                # concatenate scp files
                for n in $(seq ${nj}); do
                    cat "${outdir}/${name}/feats.$n.scp" || exit 1;
                done > ${outdir}/${name}/feats.scp
            ) &
            pids+=($!) # store background pids
            done
            i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
            [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

            echo "Synthesis..."

            pids=() # initialize pids
            for name in ${pair_valid_set} ${pair_test_set}; do
            (
                [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}

                # Normalization
                # If not using pretrained models statistics, use statistics of target speaker
                if [ -n "${pretrained_model}" ]; then
                    trg_cmvn="$(find "${db_root}/${pretrained_model}" -name "cmvn.ark" -print0 | xargs -0 ls -t | head -n 1)"
                else
                    trg_cmvn=data/${trg_train_set}/cmvn.ark
                fi
                apply-cmvn --norm-vars=true --reverse=true ${trg_cmvn} \
                    scp:${outdir}/${name}/feats.scp \
                    ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

                # GL
                if [ ${voc} = "GL" ]; then
                    echo "Using Griffin-Lim phase recovery."
                    convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                        --fs ${fs} \
                        --fmax "${fmax}" \
                        --fmin "${fmin}" \
                        --n_fft ${n_fft} \
                        --n_shift ${n_shift} \
                        --win_length "${win_length}" \
                        --n_mels ${n_mels} \
                        --iters ${griffin_lim_iters} \
                        ${outdir}_denorm/${name} \
                        ${outdir}_denorm/${name}/log \
                        ${outdir}_denorm/${name}/wav
                fi
            ) &
            pids+=($!) # store background pids
            done
            i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
            [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        fi


        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            echo "stage 6: Objective Evaluation"

            for name in ${pair_valid_set} ${pair_test_set}; do
                local/ob_eval/evaluate.sh --nj ${nj} \
                    --db_root ${db_root} \
                    --vocoder ${voc} \
                    ${outdir} ${name}
            done
        fi
    fi
fi
