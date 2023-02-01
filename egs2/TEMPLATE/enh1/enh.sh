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
stage=1          # Processes starts from the specified stage.
stop_stage=10000 # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip inference and evaluation stages
skip_upload=true     # Skip packing and uploading stages
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
inference_nj=32     # The number of parallel jobs in inference.
gpu_inference=false # Whether to perform gpu inference.
expdir=exp       # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Enhancement model related
enh_exp=    # Specify the directory path for enhancement experiment. If this option is specified, enh_tag is ignored.
enh_tag=    # Suffix to the result dir for enhancement model training.
enh_config= # Config for enhancement model training.
enh_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in enhancement config.
ref_num=2   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
inf_num=    # Number of inferences output by the model
            # Note that if it is not specified, it will be the same as ref_num. Otherwise, it will be overwritten.
            # In MixIT, number of outputs is larger than that of references.
noise_type_num=1    # Number of noise types in the input audio 
dereverb_ref_num=1  # Number of reference signals for deverberation
is_tse_task=false   # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks

# Training data related
use_dereverb_ref=false
use_noise_ref=false
extra_wav_list= # Extra list of scp files for wav formatting

# Pretrained model related
# The number of --init_param must be same.
init_param=

# Enhancement related
inference_args="--normalize_output_wav true"
inference_model=valid.loss.ave.pth
download_model=

# Evaluation related
scoring_protocol="STOI SDR SAR SIR SI_SNR"
ref_channel=0
inference_tag=  # Prefix to the result dir for ENH inference.
inference_enh_config= # Config for enhancement.
score_with_asr=false
asr_exp=""       # asr model for scoring WER
lm_exp=""       # lm model for scoring WER
inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
inference_lm=valid.loss.best.pth       # Language model path for decoding.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
inference_asr_tag=    # Suffix to the result dir for decoding.
inference_asr_config= # Config for decoding.
inference_asr_args=   # Arguments for ASR decoding, e.g., "--lm_weight 0.1".

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training
lang=noinfo      # The language type of corpus

# Upload model related
hf_repo=

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip inference and evaluation stages (default="${skip_eval}").
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


    # Enhancemnt model related
    --enh_tag    # Suffix to the result dir for enhancement model training (default="${enh_tag}").
    --enh_config # Config for enhancement model training (default="${enh_config}").
    --enh_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${enh_args}").
                 # Note that it will overwrite args in enhancement config.
    --ref_num    # Number of references for training (default="${ref_num}").
                 # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers. 
    --inf_num    # Number of inference audio generated by the model (default="${ref_num}")
                 # Note that if it is not specified, it will be the same as ref_num. Otherwise, it will be overwritten.
                 # In MixIT, number of outputs is larger than that of references.
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")
    --is_tse_task     # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks (default="${is_tse_task}")

    # Training data related
    --use_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_noise_ref}")
    --extra_wav_list   # Extra list of scp files for wav formatting (default="${extra_wav_list}")

    # Pretrained model related
    --init_param    # pretrained model path and module name (default="${init_param}")

    # Enhancement related
    --inference_args       # Arguments for enhancement in the inference stage (default="${inference_args}")
    --inference_model      # Enhancement model path for inference (default="${inference_model}").
    --inference_enh_config # Configuration file for overwriting some model attributes during SE inference. (default="${inference_enh_config}")

    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # ASR evaluation related
    --score_with_asr       # Enable ASR evaluation (default="${score_with_asr}")
    --asr_exp              # asr model for scoring WER  (default="${asr_exp}")
    --lm_exp               # lm model for scoring WER (default="${lm_exp}")
    --nlsyms_txt           # Non-linguistic symbol list if existing.  (default="${nlsyms_txt}")
    --inference_asr_model  # ASR model path for decoding. (default="${inference_asr_model}")
    --inference_lm         # Language model path for decoding. (default="${inference_lm}")
    --nlsyms_txt           # Non-linguistic symbol list if existing. (default="${nlsyms_txt}")
    --inference_asr_tag    # Suffix to the result dir for decoding. (default="${inference_asr_tag}")
    --inference_asr_config # Config for ASR decoding.  (default="${inference_asr_config}")
    --inference_asr_args   # Arguments for ASR decoding, e.g., "--lm_weight 0.1". (default="${inference_asr_args}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --enh_speech_fold_length # fold_length for speech data during enhancement training  (default="${enh_speech_fold_length}").
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

# Extra files for enhancement process
utt_extra_files="utt2category"

data_feats=${dumpdir}/raw

if $is_tse_task; then
    if $use_noise_ref; then
        log "--use_noise_ref must be false for the target speaker extraction (TSE) task"
        exit 1
    fi
    if $use_dereverb_ref; then
        log "--use_dereverb_ref must be false for the target speaker extraction (TSE) task"
        exit 1
    fi
    if [ -n "$inf_num" ] && [ "$inf_num" != "$ref_num" ]; then
        log "The value of '--inf_num' must be equal to that of '--ref_num' for the target speaker extraction (TSE) task"
        exit 1
    fi
fi
inf_num=${inf_num:=${ref_num}}

# Set tag for naming of model directory
if [ -z "${enh_tag}" ]; then
    if [ -n "${enh_config}" ]; then
        enh_tag="$(basename "${enh_config}" .yaml)_${feats_type}"
    else
        enh_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${enh_args}" ]; then
        enh_tag+="$(echo "${enh_args}" | sed -e "s/--\|\//\_/g" -e "s/[ |=]//g")"
    fi
fi

if [ -z "${inference_asr_tag}" ]; then
    if [ -n "${inference_asr_config}" ]; then
        inference_asr_tag="$(basename "${inference_asr_config}" .yaml)"
    else
        inference_asr_tag=asr_inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_asr_args}" ]; then
        inference_asr_tag+="$(echo "${inference_asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if [ -n "${lm_exp}" ]; then
        inference_asr_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_asr_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi



# The directory used for collect-stats mode
enh_stats_dir="${expdir}/enh_stats_${fs}"
# The directory used for training commands
if [ -z "${enh_exp}" ]; then
enh_exp="${expdir}/enh_${enh_tag}"
fi

if [ -n "${speed_perturb_factors}" ]; then
  enh_stats_dir="${enh_stats_dir}_sp"
  enh_exp="${enh_exp}_sp"
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_enh_config}" ]; then
        inference_tag="$(basename "${inference_enh_config}" .yaml)"
    else
        inference_tag=enhanced
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
        if ! $use_dereverb_ref && [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

            _scp_list="wav.scp "
            for i in $(seq ${ref_num}); do
                _scp_list+="spk${i}.scp "
            done

           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   scripts/utils/perturb_enh_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
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

        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

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
            for i in $(seq ${ref_num}); do
                _spk_list+="spk${i} "
                if $is_tse_task; then
                    _spk_list+="enroll_spk${i} "
                fi
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
                if ${is_tse_task} && [ "${dset}" = "${train_set}" ] && [ "${spk}" = "enroll_spk${i}" ]; then
                    audio_path=$(head -n 1 "data/${dset}/${spk}.scp" | awk '{print $2}')
                    if [ "${audio_path:0:1}" = "*" ]; then
                        # a special format in `enroll_spk?.scp`:
                        # MIXTURE_UID *UID SPEAKER_ID
                        cp "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}/${spk}.scp"
                        continue
                    fi
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --out-filename "${spk}.scp" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                    "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

            done

            for f in $extra_wav_list; do
                if [ -e "data/${dset}/$f" ]; then
                    # shellcheck disable=SC2086
                    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                        --out-filename "$f" \
                        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                        "data/${dset}/$f" "${data_feats}/${dset}" \
                        "${data_feats}/${dset}/logs/${f%.*}" "${data_feats}/${dset}/data/${f%.*}"
                fi
            done

            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            for f in ${utt_extra_files}; do
                [ -f data/${dset}/${f} ] && cp data/${dset}/${f} ${data_feats}${_suf}/${dset}/${f}
            done

        done
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            _spk_list=" "
            _scp_list=" "
            for i in $(seq ${ref_num}); do
                _spk_list+="spk${i} "
                _scp_list+="spk${i}.scp "
                if $is_tse_task; then
                    _spk_list+="enroll_spk${i} "
                    _scp_list+="enroll_spk${i}.scp "
                fi
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
            for f in ${utt_extra_files}; do
                if [ -f "${data_feats}/org/${dset}/${f}" ]; then
                    cp "${data_feats}/org/${dset}/${f}" "${data_feats}/${dset}/${f}"
                fi
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
            utils/fix_data_dir.sh --utt_extra_files "${_scp_list} ${utt_extra_files}" "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the data preparation stages"
fi


# ========================== Data preparation is done here. ==========================



if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: Enhancement collect stats: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        # 1. Split the key file
        _logdir="${enh_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_enh_train_dir}/${_scp} wc -l)" "$(<${_enh_valid_dir}/${_scp} wc -l)")

        key_file="${_enh_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_enh_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${enh_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${enh_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${enh_stats_dir}/run.sh"; chmod +x "${enh_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Enhancement collect-stats started... log: '${_logdir}/stats.*.log'"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,${_type} "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,${_type} "
        for spk in $(seq "${ref_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/spk${spk}.scp,speech_ref${spk},${_type} "

            # for target-speaker extraction
            if $is_tse_task; then
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/enroll_spk${spk}.scp,enroll_ref${spk},text "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/enroll_spk${spk}.scp,enroll_ref${spk},text "
            fi
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            _train_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "; done)
            _valid_data_param+=$(for n in $(seq $dereverb_ref_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_enh_valid_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "; done)
        fi

        if $use_noise_ref; then
            # references for denoising
            _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},${_type} "; done)
            _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
                "--valid_data_path_and_name_and_type ${_enh_valid_dir}/noise${n}.scp,noise_ref${n},${_type} "; done)
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        if $is_tse_task; then
            train_module=espnet2.bin.enh_tse_train
        else
            train_module=espnet2.bin.enh_train
        fi
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m ${train_module} \
                --collect_stats true \
                ${_train_data_param} \
                ${_valid_data_param} \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${enh_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${enh_stats_dir}"

    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: Enhancemnt Frontend Training: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp="wav.scp"
        # "sound" supports "wav", "flac", etc.
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _fold_length="$((enh_speech_fold_length * 100))"

        # prepare train and valid data parameters
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/${_scp},speech_mix,${_type} "
        _train_shape_param="--train_shape_file ${enh_stats_dir}/train/speech_mix_shape "
        _fold_length_param="--fold_length ${_fold_length} "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,${_type} "
        _valid_shape_param="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_shape "

        for spk in $(seq "${ref_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/speech_ref${spk}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "

            # for target-speaker extraction
            if $is_tse_task; then
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/enroll_spk${spk}.scp,enroll_ref${spk},text "
                _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/enroll_ref${spk}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            fi
        done

        for spk in $(seq "${ref_num}"); do
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/spk${spk}.scp,speech_ref${spk},${_type} "
            _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/speech_ref${spk}_shape "

            # for target-speaker extraction
            if $is_tse_task; then
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/enroll_spk${spk}.scp,enroll_ref${spk},text "
                _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/enroll_ref${spk}_shape "
            fi
        done

        if $use_dereverb_ref; then
            # references for dereverberation
            for n in $(seq "${dereverb_ref_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "
                _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/dereverb_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/dereverb${n}.scp,dereverb_ref${n},${_type} "
                _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/dereverb_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        if $use_noise_ref; then
            # references for denoising
            for n in $(seq "${noise_type_num}"); do
                _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/noise${n}.scp,noise_ref${n},${_type} "
                _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/noise_ref${n}_shape "
                _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/noise${n}.scp,noise_ref${n},${_type} "
                _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/noise_ref${n}_shape "
                _fold_length_param+="--fold_length ${_fold_length} "
            done
        fi

        log "Generate '${enh_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${enh_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${enh_exp}/run.sh"; chmod +x "${enh_exp}/run.sh"

        log "enh training started... log: '${enh_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${enh_exp})"
        else
            jobname="${enh_exp}/train.log"
        fi
        if $is_tse_task; then
            train_module=espnet2.bin.enh_tse_train
        else
            train_module=espnet2.bin.enh_train
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${enh_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${enh_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m ${train_module} \
                ${_train_data_param} \
                ${_valid_data_param} \
                ${_train_shape_param} \
                ${_valid_shape_param} \
                ${_fold_length_param} \
                --resume true \
                --output_dir "${enh_exp}" \
                ${init_param:+--init_param $init_param} \
                ${_opts} ${enh_args}

    fi
else
    log "Skip the training stages"
fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Enhance Speech: training_dir=${enh_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${enh_exp}/run_enhance.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${enh_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${enh_exp}/run_enhance.sh"; chmod +x "${enh_exp}/run_enhance.sh"
        _opts=

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${enh_exp}/${inference_tag}_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi

            # for target-speaker extraction
            _data_param="--data_path_and_name_and_type ${_data}/${_scp},speech_mix,${_type} "
            if $is_tse_task; then
                for spk in $(seq "${ref_num}"); do
                    _data_param+="--data_path_and_name_and_type ${_data}/enroll_spk${spk}.scp,enroll_ref${spk},text "
                done
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

            # 2. Submit inference jobs
            log "Enhancement started... log: '${_logdir}/enh_inference.*.log'"
            if $is_tse_task; then
                infer_module=espnet2.bin.enh_tse_inference
            else
                infer_module=espnet2.bin.enh_inference
            fi
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/enh_inference.JOB.log \
                ${python} -m ${infer_module} \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    ${_data_param} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${enh_exp}"/config.yaml \
                    ${inference_enh_config:+--inference_config "$inference_enh_config"} \
                    --model_file "${enh_exp}"/"${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/enh_inference.*.log) ; exit 1; }


            _spk_list=" "
            for i in $(seq ${inf_num}); do
                _spk_list+="spk${i} "
            done

            # 3. Concatenates the output files from each jobs
            for spk in ${_spk_list} ; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${spk}.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/${spk}.scp"
            done

        done
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Scoring"
        _cmd=${decode_cmd}

        # score_obs=true: Scoring for observation signal
        # score_obs=false: Scoring for enhanced signal
        for score_obs in true false; do
            # Peform only at the first time for observation
            if "${score_obs}" && [ -e "${data_feats}/RESULTS.md" ]; then
                log "${data_feats}/RESULTS.md already exists. The scoring for observation will be skipped"
                continue
            fi

            for dset in "${valid_set}" ${test_sets}; do
                _data="${data_feats}/${dset}"
                if "${score_obs}"; then
                    _dir="${data_feats}/${dset}/scoring"
                else
                    _dir="${enh_exp}/${inference_tag}_${dset}/scoring"
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
                for spk in $(seq "${ref_num}"); do
                    _ref_scp+="--ref_scp ${_data}/spk${spk}.scp "
                done
                _inf_scp=
                if "${score_obs}"; then
                    for spk in $(seq "${ref_num}"); do
                        # To compute the score of observation, input original wav.scp
                        _inf_scp+="--inf_scp ${data_feats}/${dset}/wav.scp "
                    done
                    flexible_numspk=false
                else
                    for spk in $(seq "${inf_num}"); do
                        _inf_scp+="--inf_scp ${enh_exp}/${inference_tag}_${dset}/spk${spk}.scp "
                    done
                    if [[ "${ref_num}" -ne "${inf_num}" ]]; then
                        flexible_numspk=true
                    else
                        flexible_numspk=false
                    fi
                fi

                # 2. Submit scoring jobs
                log "Scoring started... log: '${_logdir}/enh_scoring.*.log'"
                # shellcheck disable=SC2086
                ${_cmd} JOB=1:"${_nj}" "${_logdir}"/enh_scoring.JOB.log \
                    ${python} -m espnet2.bin.enh_scoring \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_ref_scp} \
                        ${_inf_scp} \
                        --ref_channel ${ref_channel} \
                        --flexible_numspk ${flexible_numspk}

                for spk in $(seq "${ref_num}"); do
                    for protocol in ${scoring_protocol} wav; do
                        for i in $(seq "${_nj}"); do
                            cat "${_logdir}/output.${i}/${protocol}_spk${spk}"
                        done | LC_ALL=C sort -k1 > "${_dir}/${protocol}_spk${spk}"
                    done
                done


                for protocol in ${scoring_protocol}; do
                    # shellcheck disable=SC2046
                    paste $(for j in $(seq ${ref_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                    awk 'BEGIN{sum=0}
                        {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                        END{printf ("%.2f\n",sum/NR)}' > "${_dir}/result_${protocol,,}.txt"
                done
            done

            ./scripts/utils/show_enh_score.sh "${_dir}/../.." > "${_dir}/../../RESULTS.md"
        done
        log "Evaluation result for observation: ${data_feats}/RESULTS.md"
        log "Evaluation result for enhancement: ${enh_exp}/RESULTS.md"

    fi
else
    log "Skip the evaluation stages"
fi

if "${score_with_asr}"; then

    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Decode with pretrained ASR model: "
        _cmd=${decode_cmd}

        _opts=
        if [ -n "${inference_asr_config}" ]; then
            _opts+="--config ${inference_asr_config} "
        fi
        if [ -n "${lm_exp}" ]; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
        fi

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        # score_obs=true: Scoring for observation signal
        # score_obs=false: Scoring for enhanced signal
        for score_obs in true false; do
            # Peform only at the first time for observation
            if "${score_obs}" && [ -e "${data_feats}/RESULTS_ASR.md" ]; then
                log "${data_feats}/RESULTS_ASR.md already exists. The scoring for observation will be skipped"
                continue
            fi

            for dset in ${valid_set} ${test_sets}; do
                _data="${data_feats}/${dset}"
                if "${score_obs}"; then
                    _dir="${data_feats}/${inference_asr_tag}/${dset}/"
                else
                    _dir="${enh_exp}/${inference_asr_tag}/${dset}"
                fi

                for spk in $(seq "${ref_num}"); do
                    _ddir=${_dir}/spk_${spk}
                    _logdir="${_ddir}/logdir"
                    _decode_dir="${_ddir}/decode"
                    mkdir -p ${_ddir}
                    mkdir -p "${_logdir}"
                    mkdir -p "${_decode_dir}"

                    if "${score_obs}"; then
                        # Using same wav.scp for all speakers
                        cp "${_data}/wav.scp" "${_ddir}/wav.scp"
                    else
                        cp "${enh_exp}/${inference_tag}_${dset}/scoring/wav_spk${spk}" "${_ddir}/wav.scp"
                    fi
                    cp data/${dset}/text_spk${spk} ${_ddir}/text
                    cp ${_data}/{spk2utt,utt2spk,utt2num_samples,feats_type} ${_ddir}
                    utils/fix_data_dir.sh "${_ddir}"
                    mv ${_ddir}/wav.scp ${_ddir}/wav_ori.scp

                    scripts/audio/format_wav_scp.sh --nj "${inference_nj}" --cmd "${_cmd}" \
                        --out-filename "wav.scp" \
                        --audio-format "${audio_format}" --fs "${fs}" \
                        "${_ddir}/wav_ori.scp" "${_ddir}" \
                        "${_ddir}/formated/logs/" "${_ddir}/formated/"

                    if [[ "${audio_format}" == *ark* ]]; then
                        _type=kaldi_ark
                    else
                        # "sound" supports "wav", "flac", etc.
                        _type=sound
                    fi

                    # 1. Split the key file
                    key_file=${_ddir}/wav.scp
                    _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

                    split_scps=""
                    for n in $(seq "${_nj}"); do
                        split_scps+=" ${_logdir}/keys.${n}.scp"
                    done
                    # shellcheck disable=SC2086
                    utils/split_scp.pl "${key_file}" ${split_scps}

                    log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
                    # shellcheck disable=SC2086
                    ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                        python3 -m espnet2.bin.asr_inference \
                            --ngpu "${_ngpu}" \
                            --data_path_and_name_and_type "${_ddir}/wav.scp,speech,${_type}" \
                            --key_file "${_logdir}"/keys.JOB.scp \
                            --asr_train_config "${asr_exp}"/config.yaml \
                            --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                            --output_dir "${_logdir}"/output.JOB \
                            ${_opts} ${inference_asr_args}


                    for f in token token_int score text; do
                        for i in $(seq "${_nj}"); do
                            cat "${_logdir}/output.${i}/1best_recog/${f}"
                        done | LC_ALL=C sort -k1 >"${_decode_dir}/${f}"
                    done
                done
            done
        done
    fi

    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: Scoring with pretrained ASR model: "

        _cmd=${decode_cmd}
        cleaner=none

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        # score_obs=true: Scoring for observation signal
        # score_obs=false: Scoring for enhanced signal
        for score_obs in true false; do
            # Peform only at the first time for observation
            if "${score_obs}" && [ -e "${data_feats}/RESULTS_ASR.md" ]; then
                log "${data_feats}/RESULTS_ASR.md already exists. The scoring for observation will be skipped"
                continue
            fi

            for dset in ${valid_set} ${test_sets}; do
                if "${score_obs}"; then
                    _dir="${data_feats}/${inference_asr_tag}/${dset}"
                else
                    _dir="${enh_exp}/${inference_asr_tag}/${dset}/"
                fi

                for spk in $(seq "${ref_num}"); do
                    _ddir=${_dir}/spk_${spk}
                    _logdir="${_ddir}/logdir"
                    _decode_dir="${_ddir}/decode"

                    for _type in cer wer; do

                        _scoredir="${_ddir}/score_${_type}"
                        mkdir -p "${_scoredir}"

                        if [ "${_type}" = wer ]; then
                            # Tokenize text to word level
                            paste \
                                <(<"${_ddir}/text" \
                                    python3 -m espnet2.bin.tokenize_text  \
                                        -f 2- --input - --output - \
                                        --token_type word \
                                        --non_linguistic_symbols "${nlsyms_txt}" \
                                        --remove_non_linguistic_symbols true \
                                        --cleaner "${cleaner}" \
                                        ) \
                                <(<"${_ddir}/text" awk '{ print "(" $1 ")" }') \
                                    >"${_scoredir}/ref.trn"

                            # NOTE(kamo): Don't use cleaner for hyp
                            paste \
                                <(<"${_decode_dir}/text"  \
                                    python3 -m espnet2.bin.tokenize_text  \
                                        -f 2- --input - --output - \
                                        --token_type word \
                                        --non_linguistic_symbols "${nlsyms_txt}" \
                                        --remove_non_linguistic_symbols true \
                                        ) \
                                <(<"${_ddir}/text" awk '{ print "(" $1 ")" }') \
                                    >"${_scoredir}/hyp.trn"
                        elif [ "${_type}" = cer ]; then
                            # Tokenize text to char level
                            paste \
                                <(<"${_ddir}/text" \
                                    python3 -m espnet2.bin.tokenize_text  \
                                        -f 2- --input - --output - \
                                        --token_type char \
                                        --non_linguistic_symbols "${nlsyms_txt}" \
                                        --remove_non_linguistic_symbols true \
                                        --cleaner "${cleaner}" \
                                        ) \
                                <(<"${_ddir}/text" awk '{ print "(" $1 ")" }') \
                                    >"${_scoredir}/ref.trn"

                            # NOTE(kamo): Don't use cleaner for hyp
                            paste \
                                <(<"${_decode_dir}/text"  \
                                    python3 -m espnet2.bin.tokenize_text  \
                                        -f 2- --input - --output - \
                                        --token_type char \
                                        --non_linguistic_symbols "${nlsyms_txt}" \
                                        --remove_non_linguistic_symbols true \
                                        ) \
                                <(<"${_ddir}/text" awk '{ print "(" $1 ")" }') \
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
            done

            scripts/utils/show_asr_result.sh "${_dir}/../../" > "${_dir}"/../../RESULTS_ASR.md
        done
        log "Evaluation result for observation: ${data_feats}/RESULTS_ASR.md"
        log "Evaluation result for enhancement: ${enh_exp}/RESULTS_ASR.md"
    fi
else
    log "Skip the stages for scoring with asr"
fi



packed_model="${enh_exp}/${enh_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 11: Pack model: ${packed_model}"

        ${python} -m espnet2.bin.pack enh \
            --train_config "${enh_exp}"/config.yaml \
            --model_file "${enh_exp}"/"${inference_model}" \
            --option "${enh_exp}"/RESULTS.md \
            --option "${enh_stats_dir}"/train/feats_stats.npz  \
            --option "${enh_exp}"/images \
            --outpath "${packed_model}"
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Upload model to Zenodo: ${packed_model}"
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
        # /some/where/espnet/egs2/foo/asr1/ -> foo/asr1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/asr1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${enh_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${enh_exp}"/RESULTS.md)</code></pre></li>
<li><strong>ASR config</strong><pre><code>$(cat "${enh_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${enh_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace" && \
            exit 1
        log "Stage 13: Upload model to HuggingFace: ${hf_repo}"

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
        hf_task=audio-to-audio
        # shellcheck disable=SC2034
        espnet_task=ENH
        # shellcheck disable=SC2034
        task_exp=${enh_exp}
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
