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
stop_stage=10000 # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
inference_nj=32     # The number of parallel jobs in decoding.
gpu_inference=false # Whether to perform gpu decoding.
expdir=exp       # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Diarization model related
diarization_tag=    # Suffix to the result dir for diarization model training.
diarization_config= # Config for diarization model training.
diarization_args=   # Arguments for diarization model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in diarization config.
spk_num=2


# Diarization related
inference_args="--normalize_output_wav true"
inference_model=valid.acc.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
diarization_speech_fold_length=800 # fold_length for speech data during diarization training
lang=noinfo      # The language type of corpus


help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu          # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes     # The number of nodes
    --nj            # The number of parallel jobs (default="${nj}").
    --inference_nj  # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir       # Directory to dump features (default="${dumpdir}").
    --expdir        # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (only support raw currently).
    --audio_format # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").


    # Diarization model related
    --diarization_tag    # Suffix to the result dir for diarization model training (default="${diarization_tag}").
    --diarization_config # Config for diarization model training (default="${diarization_config}").
    --diarization_args   # Arguments for diarization model training, e.g., "--max_epoch 10" (default="${diarization_args}").
                 # Note that it will overwrite args in diarization config.
    --spk_num    # Number of speakers in the input audio (default="${spk_num}")

    # diarization related
    --inference_args      # Arguments for diarization in the inference stage (default="${inference_args}")
    --inference_model # diarization model path for inference (default="${inference_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --diarization_speech_fold_length # fold_length for speech data during diarization training  (default="${diarization_speech_fold_length}").
    --lang         # The language type of corpus (default="${lang}")
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
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw

# Set tag for naming of model directory
if [ -z "${diarization_tag}" ]; then
    if [ -n "${diarization_config}" ]; then
        diarization_tag="$(basename "${diarization_config}" .yaml)_${feats_type}"
    else
        diarization_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${diarization_args}" ]; then
        diarization_tag+="$(echo "${diarization_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for collect-stats mode
diarization_stats_dir="${expdir}/diarization_stats_${fs}"
# The directory used for training commands
diarization_exp="${expdir}/diarization_${diarization_tag}"

if [ -n "${speed_perturb_factors}" ]; then
  diarization_stats_dir="${diarization_stats_dir}_sp"
  diarization_exp="${diarization_exp}_sp"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

        log "Stage 2: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

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
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi


            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done

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


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for spk in ${_spk_list};do
                cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
            done

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            for spk in ${_spk_list} "wav"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================



