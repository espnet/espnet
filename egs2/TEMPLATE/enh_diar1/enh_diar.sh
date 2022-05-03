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
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=8k                # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second
max_wav_duration=    # Maximum duration in second

# diar model related
diar_tag=    # Suffix to the result dir for diar enh model training.
diar_config= # Config for diar model training.
diar_args=   # Arguments for diar model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in diar config.
feats_normalize=utterance_mvn # Normalizaton layer type.
spk_num=2    # Number of speakers in the input audio
noise_type_num=1
dereverb_ref_num=1

# Training data related
use_dereverb_ref=false
use_noise_ref=false

# diar related
inference_config= # Config for diar model inference
inference_model=valid.si_snr_loss.best.pth
inference_tag=    # Suffix to the inference dir for diar model inference
download_model=   # Download a model from Model Zoo and use it for diarization.

# Upload model related
hf_repo=

# diar scoring related
collar=0         # collar for der scoring
frame_shift=64  # frame shift to convert frame-level label into real time
                 # this should be aligned with frontend feature extraction

# enh Evaluation related
scoring_protocol="STOI SDR SAR SIR SI_SNR"
ref_channel=0

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
diar_speech_fold_length=800 # fold_length for speech data during diar training
                            # Typically, the label also follow the same fold length
lang=noinfo      # The language type of corpus.


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

    # Feature extraction related
    --feats_type       # Feature type (only support raw currently).
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").


    # Diarization model related
    --diar_tag        # Suffix to the result dir for diarization model training (default="${diar_tag}").
    --diar_config     # Config for diarization model training (default="${diar_config}").
    --diar_args       # Arguments for diarization model training, e.g., "--max_epoch 10" (default="${diar_args}").
                      # Note that it will overwrite args in diar config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --spk_num         # Number of speakers in the input audio (default="${spk_num}")

    # Diarization related
    --inference_config # Config for diar model inference
    --inference_model  # diarization model path for inference (default="${inference_model}").
    --inference_tag    # Suffix to the inference dir for diar model inference
    --download_model   # Download a model from Model Zoo and use it for diarization (default="${download_model}").

    # Scoring related
    --collar      # collar for der scoring
    --frame_shift # frame shift to convert frame-level label into real time

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set               # Name of training set (required).
    --valid_set               # Name of development set (required).
    --test_sets               # Names of evaluation sets (required).
    --diar_speech_fold_length # fold_length for speech data during diarization training  (default="${diar_speech_fold_length}").
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
if [ -z "${diar_tag}" ]; then
    if [ -n "${diar_config}" ]; then
        diar_tag="$(basename "${diar_config}" .yaml)_${feats_type}"
    else
        diar_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${diar_args}" ]; then
        diar_tag+="$(echo "${diar_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
fi

# The directory used for collect-stats mode
diar_stats_dir="${expdir}/diar_enh_stats_${fs}"
# The directory used for training commands
diar_exp="${expdir}/diar_enh_${diar_tag}"

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
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{wav.scp,reco2file_and_channel}

            # shellcheck disable=SC2086
            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done
            if $use_noise_ref && [ -n "${_suf}" ]; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
            fi
            if $use_dereverb_ref && [ -n "${_suf}" ]; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
            fi

            for spk in ${_spk_list} "wav" ; do
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --out-filename "${spk}.scp" \
                    --audio-format "${audio_format}" --fs "${fs}" \
                    "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                    "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

            done
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            # specifics for diarization
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}${_suf}/${dset}"/utt2spk \
                    "${data_feats}${_suf}/${dset}"/segments \
                    "${data_feats}${_suf}/${dset}"/rttm

            # convert standard rttm file into espnet-format rttm (measure with samples)
            pyscripts/utils/convert_rttm.py \
                --rttm "${data_feats}${_suf}/${dset}"/rttm \
                --wavscp "${data_feats}${_suf}/${dset}"/wav.scp \
                --output_path "${data_feats}${_suf}/${dset}" \
                --sampling_rate "${fs}"
        done
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            _spk_list=" "
            _scp_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
                _scp_list+="spk${i}.scp "
            done
            if $use_noise_ref; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                _scp_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n.scp "; done)
            fi
            if $use_dereverb_ref; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                _scp_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n.scp "; done)
            fi

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for spk in ${_spk_list};do
                cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
            done

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            # diarization typically accept long recordings, so does not has
            # max length requirements
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" \
                    '{ if ($2 > min_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            for spk in ${_spk_list} "wav"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            #utils/fix_data_dir.sh --utt_extra_files "${_scp_list}" "${data_feats}/${dset}"
            utils/fix_data_dir.sh "${data_feats}/${dset}"
            #sort spk{i}.scp, etc.
            #for spk in ${_spk_list} ; do
            #    sort -t '-' "${data_feats}/${dset}/${spk}.scp" -o "${data_feats}/${dset}/${spk}.scp"
            #done            

            # specifics for diarization
            steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    "${data_feats}/${dset}"/utt2spk \
                    "${data_feats}/${dset}"/segments \
                    "${data_feats}/${dset}"/rttm

            # convert standard rttm file into espnet-format rttm (measure with samples)
            pyscripts/utils/convert_rttm.py \
                --rttm "${data_feats}/${dset}"/rttm \
                --wavscp "${data_feats}/${dset}"/wav.scp \
                --output_path "${data_feats}/${dset}" \
                --sampling_rate "${fs}"
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _diar_train_dir="${data_feats}/${train_set}"
        _diar_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: Diarization Enhancement collect stats: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

        _opts=
        if [ -n "${diar_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.diar_enh_train --print_config --optim adam
            _opts+="--config ${diar_config} "
        fi

        _feats_type="$(<${_diar_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        _opts+="--diar_num_spk ${spk_num} "

        # 1. Split the key file
        _logdir="${diar_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_diar_train_dir}/${_scp} wc -l)" "$(<${_diar_valid_dir}/${_scp} wc -l)")

        key_file="${_diar_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_diar_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${diar_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${diar_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${diar_stats_dir}/run.sh"; chmod +x "${diar_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Diarization Enhancement collect-stats started... log: '${_logdir}/stats.*.log'"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_diar_train_dir}/${_scp},speech,${_type} "
        _valid_data_param="--valid_data_path_and_name_and_type ${_diar_valid_dir}/${_scp},speech,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/espnet_rttm,text,rttm "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/espnet_rttm,text,rttm "
        for spk in $(seq "${spk_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            _train_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--train_data_path_and_name_and_type ${_diar_train_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "; done)
            _valid_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_diar_valid_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "; done)
        fi

        if $use_noise_ref; then
            # references for denoising
            _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--train_data_path_and_name_and_type ${_diar_train_dir}/noise${n}.scp,noise_ref${n},${_type} "; done)
            _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_diar_valid_dir}/noise${n}.scp,noise_ref${n},${_type} "; done)
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.enh_s2t_train \
                --collect_stats true \
                --use_preprocessor true \
                ${_train_data_param} \
                ${_valid_data_param} \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${diar_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${diar_stats_dir}"

    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _diar_train_dir="${data_feats}/${train_set}"
        _diar_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: Diarization Enhancement Training: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

        _opts=
        if [ -n "${diar_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
            _opts+="--config ${diar_config} "
        fi

        _feats_type="$(<${_diar_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((diar_speech_fold_length * 100))"
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${diar_stats_dir}/train/feats_stats.npz "
        fi

        _opts+="--diar_num_spk ${spk_num} "

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_diar_train_dir}/wav.scp,speech,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/espnet_rttm,text,rttm "
        _train_shape_param="--train_shape_file ${diar_stats_dir}/train/speech_shape "
        _train_shape_param+="--train_shape_file ${diar_stats_dir}/train/text_shape "
        _valid_data_param="--valid_data_path_and_name_and_type ${_diar_valid_dir}/wav.scp,speech,${_type} "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/espnet_rttm,text,rttm "
        _valid_shape_param="--valid_shape_file ${diar_stats_dir}/valid/speech_shape "
        _valid_shape_param+="--valid_shape_file ${diar_stats_dir}/valid/text_shape "
        _fold_length_param="--fold_length ${_fold_length} "
        for spk in $(seq "${spk_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _train_shape_param+="--train_shape_file ${diar_stats_dir}/train/speech_ref${spk}_shape "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _valid_shape_param+="--valid_shape_file ${diar_stats_dir}/valid/speech_ref${spk}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            for n in $(seq "${dereverb_ref_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "
                _train_shape_param+="--train_shape_file ${diar_stats_dir}/train/dereverb_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "
                _valid_shape_param+="--valid_shape_file ${diar_stats_dir}/valid/dereverb_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        if $use_noise_ref; then
            # references for denoising
            for n in $(seq "${noise_type_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_diar_train_dir}/noise${n}.scp,noise_ref${n},${_type} "
                _train_shape_param+="--train_shape_file ${diar_stats_dir}/train/noise_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/noise${n}.scp,noise_ref${n},${_type} "
                _valid_shape_param+="--valid_shape_file ${diar_stats_dir}/valid/noise_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        log "Generate '${diar_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${diar_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${diar_exp}/run.sh"; chmod +x "${diar_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "Diarization training started... log: '${diar_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${diar_exp})"
        else
            jobname="${diar_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${diar_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${diar_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.enh_s2t_train \
                --use_preprocessor true \
                --resume true \
                --fold_length "${diar_speech_fold_length}" \
                --output_dir "${diar_exp}" \
                ${_train_data_param} \
                ${_valid_data_param} \
                ${_train_shape_param} \
                ${_valid_shape_param} \
                ${_fold_length_param} \
                ${_opts} ${diar_args}

    fi
else
    log "Skip the training stages"
fi

if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    diar_exp="${expdir}/${download_model}"
    mkdir -p "${diar_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${diar_exp}/config.txt"

    # Get the path of each file
    _diar_model_file=$(<"${diar_exp}/config.txt" sed -e "s/.*'diar_model_file': '\([^']*\)'.*$/\1/")
    _diar_train_config=$(<"${diar_exp}/config.txt" sed -e "s/.*'diar_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_diar_model_file}" "${diar_exp}"
    ln -sf "${_diar_train_config}" "${diar_exp}"
    inference_diar_model=$(basename "${_diar_model_file}")

fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Diarize Speaker & Enhance Speech: training_dir=${diar_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${diar_exp}/run_diar_enh.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${diar_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${diar_exp}/run_diarize.sh"; chmod +x "${diar_exp}/run_diarize.sh"
        _opts=

        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${diar_exp}/diarized_enhanced_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            _type=sound

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit inference jobs
            log "Diarization & Enhancement started... log: '${_logdir}/diar_enh_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/diar_inference.JOB.log \
                ${python} -m espnet2.bin.diar_inference \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech_mix,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${diar_exp}"/config.yaml \
                    --model_file "${diar_exp}"/"${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts}

            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done

            # 3. Concatenates the output files from each jobs
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/diarize.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/diarize.scp"

            for spk in ${_spk_list} ;
            do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${spk}.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/${spk}.scp"
            done

        done
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Diarization Scoring"
        _cmd=${decode_cmd}

        # Diarization Scoring
        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _inf_dir="${diar_exp}/diarized_enhanced_${dset}"
            _dir="${diar_exp}/diarized_enhanced_${dset}/scoring"
            mkdir -p "${_dir}"

            scripts/utils/score_der.sh ${_dir} ${_inf_dir}/diarize.scp ${_data}/rttm \
                --collar ${collar} --fs ${fs} --frame_shift ${frame_shift}
        done

        # Show results in Markdown syntax
        scripts/utils/show_diar_result.sh "${diar_exp}" > "${diar_exp}"/DIAR_RESULTS.md
        cat "${diar_exp}"/DIAR_RESULTS.md

        log "Evaluation result for diarization: ${diar_exp}/DIAR_RESULTS.md"
    fi

    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Enhancement Scoring"
        _cmd=${decode_cmd}
        # Enhancement Scoring
        # score_obs=true: Scoring for observation signal
        # score_obs=false: Scoring for enhanced signal
        for score_obs in true false; do
            # Peform only at the first time for observation
            if "${score_obs}" && [ -e "${data_feats}/RESULTS.md" ]; then
                log "${data_feats}/RESULTS.md already exists. The scoring for observation will be skipped"
                continue
            fi

            for dset in ${test_sets}; do
                _data="${data_feats}/${dset}"
                if "${score_obs}"; then
                    _dir="${data_feats}/${dset}/scoring"
                else
                    _dir="${diar_exp}/diarized_enhanced_${dset}/scoring"
                fi

                _logdir="${_dir}/logdir"
                mkdir -p "${_logdir}"

                # 1. Split the key file
                key_file=${_data}/wav.scp
                split_scps=""
                _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
                for n in $(seq "${_nj}"); do
                    split_scps+=" ${_logdir}/keys.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                _ref_scp=
                for spk in $(seq "${spk_num}"); do
                    _ref_scp+="--ref_scp ${_data}/spk${spk}.scp "
                done
                _inf_scp=
                for spk in $(seq "${spk_num}"); do
                    if "${score_obs}"; then
                        # To compute the score of observation, input original mixture in wav.scp
                        # use only the lines in wav.scp corresponding to the lines in spk${spk}.scp
                        grep -o '^.\+\s' ${_data}/spk${spk}.scp > ${_data}/utt_list
                        grep -f ${_data}/utt_list ${_data}/wav.scp > ${_data}/wav_eval${spk}.scp
                        _inf_scp+="--inf_scp ${_data}/wav_eval${spk}.scp "
                    else
                        _inf_scp+="--inf_scp ${diar_exp}/diarized_enhanced_${dset}/spk${spk}.scp "
                    fi
                done

                # 2. Submit scoring jobs
                log "Scoring started... log: '${_logdir}/enh_scoring.*.log'"
                # shellcheck disable=SC2086
                ${_cmd} JOB=1:"${_nj}" "${_logdir}"/enh_scoring.JOB.log \
                    ${python} -m espnet2.bin.enh_scoring_flexible_numspk \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_ref_scp} \
                        ${_inf_scp} \
                        --ref_channel ${ref_channel}

                for spk in $(seq "${spk_num}"); do
                    for protocol in ${scoring_protocol} wav; do
                        for i in $(seq "${_nj}"); do
                            cat "${_logdir}/output.${i}/${protocol}_spk${spk}"
                        done | LC_ALL=C sort -k1 > "${_dir}/${protocol}_spk${spk}"
                    done
                done


                for protocol in ${scoring_protocol}; do
                    # shellcheck disable=SC2046
                    paste $(for j in $(seq ${spk_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                    awk 'BEGIN{sum=0}
                        {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                        END{printf ("%.4f\n",sum/NR)}' > "${_dir}/result_${protocol,,}.txt"
                done
            done

            scripts/utils/show_enh_score.sh "${_dir}/../.." > "${_dir}/../../ENH_RESULTS.md"
            cat 
        done
        log "Evaluation result for observation: ${data_feats}/RESULTS.md"
        log "Evaluation result for enhancement: ${diar_exp}/ENH_RESULTS.md"
        cat "${diar_exp}"/ENH_RESULTS.md
    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${diar_exp}/${diar_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Pack model: ${packed_model}"

        ${python} -m espnet2.bin.pack diar \
            --train_config "${diar_exp}"/config.yaml \
            --model_file "${diar_exp}"/"${inference_model}" \
            --option "${diar_exp}"/DIAR_RESULTS.md \
            --option "${diar_exp}"/ENH_RESULTS.md \
            --option "${diar_stats_dir}"/train/feats_stats.npz  \
            --option "${diar_exp}"/images \
            --outpath "${packed_model}"
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 9: Upload model to Zenodo: ${packed_model}"
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
        # /some/where/espnet/egs2/foo/diar1/ -> foo/diar1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/diar1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${diar_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${diar_exp}"/RESULTS.md)</code></pre></li>
<li><strong>Diarization config</strong><pre><code>$(cat "${diar_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}" \
            --description_file "${diar_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 10: Upload model to HuggingFace: ${hf_repo}"

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
        hf_task=diarization
        # shellcheck disable=SC2034
        espnet_task=DIAR
        # shellcheck disable=SC2034
        task_exp=${diar_exp}
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
