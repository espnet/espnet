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
mem=10G          # Memory per CPU
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
infernece_nj=32     # The number of parallel jobs in decoding.
gpu_inference=false # Whether to perform gpu decoding.
expdir=exp       # Directory to save experiments.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format (only in feats_type=raw).
fs=16k            # Sampling rate.

# Enhancement model related
enh_tag=    # Suffix to the result dir for enhancement model training.
enh_config= # Config for ehancement model training.
enh_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in enhancement config.
spk_num=2
noise_type_num=1
feats_normalize=global_mvn  # Normalizaton layer type

# Training data related
use_dereverb_ref=false
use_noise_ref=false

# Enhancement related
inference_args="--normalize_output_wav false"
inference_enh_model=valid.si_snr.best.pth

# Evaluation related
scoring_protocol="PESQ STOI SDR SAR SIR"
ref_channel=0

# [Task dependent] Set the datadir name created by local/data.sh
train_set=     # Name of training set.
dev_set=       # Name of development set.
eval_sets=     # Names of evaluation sets. Multiple items can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --dev-set <dev_set_name> --eval_sets <eval_set_names> 

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu          # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes     # The number of nodes
    --nj            # The number of parallel jobs (default="${nj}").
    --infernece_nj  # The number of parallel jobs in inference (default="${infernece_nj}").
    --gpu_inference # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir       # Directory to dump features (default="${dumpdir}").
    --expdir        # Directory to save experiments (default="${expdir}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").


    # Enhancemnt model related
    --enh_tag    # Suffix to the result dir for enhancement model training (default="${enh_tag}").
    --enh_config # Config for enhancement model training (default="${enh_config}").
    --enh_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${enh_args}").
                 # Note that it will overwrite args in enhancement config.
    --spk_num    # Number of speakers in the input audio (default="${spk_num}")
    --noise_type_num  # Number of noise types in the input audio (default="${noise_type_num}")
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").

    # Training data related
    --use_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_noise_ref}")

    # Enhancement related
    --inference_args      # Arguments for enhancement in the inference stage (default="${inference_args}")
    --inference_enh_model # Enhancement model path for inference (default="${inference_enh_model}").

    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --dev_set       # Name of development set (required).
    --eval_sets     # Names of evaluation sets (required).
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
[ -z "${dev_set}" ] &&   { log "${help_message}"; log "Error: --dev_set is required"  ; exit 2; };
[ -z "${eval_sets}" ] && { log "${help_message}"; log "Error: --eval_sets is required"; exit 2; };

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





# Set tag for naming of model directory
if [ -z "${enh_tag}" ]; then
    if [ -n "${enh_config}" ]; then
        enh_tag="$(basename "${enh_config}" .yaml)_${feats_type}"
    else
        enh_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${enh_args}" ]; then
        enh_tag+="$(echo "${enh_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi


# The directory used for collect-stats mode
enh_stats_dir="${expdir}/enh_stats_${fs}"
# The directory used for training commands
enh_exp="${expdir}/enh_${enh_tag}"

if [ -n "${speed_perturb_factors}" ]; then
  enh_stats_dir="${enh_stats_dir}_sp"
  enh_exp="${enh_exp}_sp"
fi

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ -n "${speed_perturb_factors}" ]; then
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

            
            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done
            if $use_noise_ref; then
                # reference for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
            fi
            if $use_dereverb_ref; then
                # reference for dereverberation
                _spk_list+="dereverb "
            fi

            for spk in ${_spk_list} "wav" ; do
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --out-filename "${spk}.scp" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/${spk}.scp" "${data_feats}/org/${dset}" \
                    "${data_feats}/org/${dset}/logs/${spk}" "${data_feats}/org/${dset}/data/${spk}"

            done
            echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"

        done
        
    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

    for dset in "${train_set}" "${dev_set}"; do
    # NOTE: Not applying to eval_sets to keep original data

        _spk_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
        done
        if $use_noise_ref; then
            # reference for denoising ("noise1 noise2 ... niose${noise_type_num} ")
            _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
        fi
        if $use_dereverb_ref; then
            # reference for dereverberation
            _spk_list+="dereverb "
        fi

        # Copy data dir
        utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
        for spk in ${_spk_list};do
            cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
        done
        # Remove short utterances
        _feats_type="$(<${data_feats}/${dset}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            min_length=2560

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="$min_length" '{ if ($2 > min_length) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
            for spk in ${_spk_list} "wav"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done
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

        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh "${data_feats}/${dset}"
    done
fi


# ========================== Data preparation is done here. ==========================



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    _enh_train_dir="${data_feats}/${train_set}"
    _enh_dev_dir="${data_feats}/${dev_set}"
    log "Stage 5: Enhancement collect stats: train_set=${_enh_train_dir}, dev_set=${_enh_dev_dir}"

    _opts=
    if [ -n "${enh_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
        _opts+="--config ${enh_config} "
    fi

    _feats_type="$(<${_enh_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        # _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _input_size="$(<${_enh_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "
    fi

    # 1. Split the key file
    _logdir="${enh_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_enh_train_dir}/${_scp} wc -l)" "$(<${_enh_dev_dir}/${_scp} wc -l)")

    key_file="${_enh_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_enh_dev_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    log "Enhancement collect-stats started... log: '${_logdir}/stats.*.log'"        

    # prepare train and valid data parameters
    _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,sound "
    _valid_data_param="--valid_data_path_and_name_and_type ${_enh_dev_dir}/wav.scp,speech_mix,sound "
    for spk in $(seq "${spk_num}"); do
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_dev_dir}/spk${spk}.scp,speech_ref${spk},sound "
    done

    if $use_dereverb_ref; then
        # reference for dereverberation
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb.scp,dereverb_ref,sound "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_dev_dir}/dereverb.scp,dereverb_ref,sound "
    fi

    if $use_noise_ref; then
        # reference for denoising
        _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
            "--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},sound "; done)
        _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
            "--valid_data_path_and_name_and_type ${_enh_dev_dir}/noise${n}.scp,noise_ref${n},sound "; done)
    fi

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.


    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python3 -m espnet2.bin.enh_train \
            --collect_stats true \
            --use_preprocessor true \
            ${_train_data_param} \
            ${_valid_data_param} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${enh_args} \
            --batch_type unsorted

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${enh_stats_dir}"

fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    _enh_train_dir="${data_feats}/${train_set}"
    _enh_dev_dir="${data_feats}/${dev_set}"
    log "Stage 6: Enhancemnt Frontend Training: train_set=${_enh_train_dir}, dev_set=${_enh_dev_dir}"

    _opts=
    if [ -n "${enh_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
        _opts+="--config ${enh_config} "
    fi

    _feats_type="$(<${_enh_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        # "sound" supports "wav", "flac", etc.
        _type=sound
        _fold_length="$((enh_speech_fold_length * 100))"
        # _opts+="--frontend_conf fs=${fs} "
    else
        _scp=feats.scp
        _type=kaldi_ark
        _fold_length="${enh_speech_fold_length}"
        _input_size="$(<${_enh_train_dir}/feats_dim)"
        _opts+="--input_size=${_input_size} "

    fi

    # prepare train and valid data parameters
    _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,sound "
    _train_shape_param="--train_shape_file ${enh_stats_dir}/train/speech_mix_shape " 
    _valid_data_param="--valid_data_path_and_name_and_type ${_enh_dev_dir}/wav.scp,speech_mix,sound "
    _valid_shape_param="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_shape "
    _fold_length_param="--fold_length ${_fold_length} "
    for spk in $(seq "${spk_num}"); do
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/speech_ref${spk}_shape " 
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_dev_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/speech_ref${spk}_shape "
        _fold_length_param+="--fold_length ${_fold_length} "
    done

    if $use_dereverb_ref; then
        # reference for dereverberation
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb.scp,dereverb_ref,sound "
        _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/dereverb_ref_shape "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_dev_dir}/dereverb.scp,dereverb_ref,sound "
        _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/dereverb_ref_shape "
        _fold_length_param+="--fold_length ${_fold_length} "
    fi

    if $use_noise_ref; then
        # reference for denoising
        for n in $(seq "${noise_type_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},sound "
            _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/noise_ref${n}_shape "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_dev_dir}/noise${n}.scp,noise_ref${n},sound "
            _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/noise_ref${n}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "
        done
    fi


    log "enh training started... log: '${enh_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${enh_exp}/train.log --mem ${mem}" \
        --log "${enh_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${enh_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.enh_train \
            ${_train_data_param} \
            ${_valid_data_param} \
            ${_train_shape_param} \
            ${_valid_shape_param} \
            ${_fold_length_param} \
            --resume true \
            --output_dir "${enh_exp}" \
            ${_opts} ${enh_args}

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Enhance Speech: training_dir=${enh_exp}"

    if ${gpu_inference}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${enh_exp}/separate_${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _scp=wav.scp
        _type=sound

        # 1. Split the key file
        key_file=${_data}/${_scp}
        split_scps=""
        _nj=$(min "${infernece_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Separation started... log: '${_logdir}/enh_inference.*.log'"
        # shellcheck disable=SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/enh_inference.JOB.log \
            python3 -m espnet2.bin.enh_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/${_scp},speech_mix,${_type}" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --enh_train_config "${enh_exp}"/config.yaml \
                --enh_model_file "${enh_exp}"/"${inference_enh_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${inference_args}


        _spk_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
        done

        # 3. Concatenates the output files from each jobs
        for spk in ${_spk_list} ;
        do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/${spk}.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/${spk}.scp"
        done

    done
fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Scoring"
    _cmd=${decode_cmd}
    
    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _inf_dir="${enh_exp}/separate_${dset}" 
        _dir="${enh_exp}/separate_${dset}/scoring" 
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        key_file=${_data}/wav.scp
        split_scps=""
        _nj=$(min "${infernece_nj}" "$(<${key_file} wc -l)")
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
            _inf_scp+="--inf_scp ${_inf_dir}/spk${spk}.scp "
        done

        # 2. Submit decoding jobs
        log "Scoring started... log: '${_logdir}/enh_scoring.*.log'"
        # shellcheck disable=SC2086
        # ${_cmd} JOB=1:"${_nj}" "${_logdir}"/enh_scoring.JOB.log \
        #     python3 -m espnet2.bin.enh_scoring \
        #         --key_file "${_logdir}"/keys.JOB.scp \
        #         --output_dir "${_logdir}"/output.JOB \
        #         ${_ref_scp} \
        #         ${_inf_scp} \
        #         --ref_channel ${ref_channel}

        for spk in $(seq "${spk_num}"); do
            for protocol in ${scoring_protocol}; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${protocol}_spk${spk}"
                done | LC_ALL=C sort -k1 > "${_dir}/${protocol}_spk${spk}"
            done
        done

        for protocol in ${scoring_protocol}; do
            # shellcheck disable=SC2046
            echo ${protocol}: $(paste $(for j in $(seq ${spk_num}); do echo ${_dir}/${protocol}_spk${j} ; done)  | 
            awk 'BEIGN{sum=0}
                {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                END{print sum/NR}') 
        done > ${_dir}/result.txt

        cat ${_dir}/result.txt
    done

    for dset in "${dev_set}" ${eval_sets} ; do
         _dir="${enh_exp}/separate_${dset}/scoring" 
         echo "======= Results in ${dset} ======="
         cat ${_dir}/result.txt
    done > ${enh_exp}/RESULTS.TXT
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "[Option] Stage 9: Pack model: ${enh_exp}/packed.tgz"

    _opts=
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${enh_stats_dir}/train/feats_stats.npz "
    fi

    # shellcheck disable=SC2086
    python -m espnet2.bin.pack enh \
        --train_config.yaml "${enh_exp}"/config.yaml \
        --model_file.pth "${enh_exp}"/"${inference_enh_model}" \
        ${_opts} \
        --option "${enh_exp}"/RESULTS.TXT \
        --outpath "${enh_exp}/packed.tgz"

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
