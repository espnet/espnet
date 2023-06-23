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
inference_nj=1       # The number of parallel jobs in decoding.
gpu_inference=true  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.
post_process_local_data_opts= # The options given to local/data.sh for additional processing in stage 4.
auxiliary_data_tags= # the names of training data for auxiliary tasks

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# VAD model related
vad_tag=       # Suffix to the result dir for vad model training.
vad_exp=       # Specify the directory path for VAD experiment.
               # If this option is specified, vad_tag is ignored.
vad_stats_dir= # Specify the directory path for VAD statistics.
vad_config=    # Config for vad model training.
vad_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in vad config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_ref=1   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
num_inf=    # Number of inferences output by the model
            # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.
            # In MixIT, number of outputs is larger than that of references.

# Upload model related
hf_repo=

# Decoding related
use_streaming=false # Whether to use streaming decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_vad_model=valid.acc.ave.pth # VAD model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
local_score_opts=          # The options given to local/score.sh.
vad_speech_fold_length=800 # fold_length for speech data during VAD training.
vad_text_fold_length=150   # fold_length for text data during VAD training.

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

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # VAD model related
    --vad_tag          # Suffix to the result dir for vad model training (default="${vad_tag}").
    --vad_exp          # Specify the directory path for VAD experiment.
                       # If this option is specified, vad_tag is ignored (default="${vad_exp}").
    --vad_stats_dir    # Specify the directory path for VAD statistics (default="${vad_stats_dir}").
    --vad_config       # Config for vad model training (default="${vad_config}").
    --vad_args         # Arguments for vad model training (default="${vad_args}").
                       # e.g., --vad_args "--max_epoch 10"
                       # Note that it will overwrite args in vad config.
    --pretrained_model=          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_ref    # Number of references for training (default="${num_ref}").
                 # In supervised learning based speech recognition, it is equivalent to number of speakers.
    --num_inf    # Number of inference audio generated by the model (default="${num_inf}")
                 # Note that if it is not specified, it will be the same as num_ref. Otherwise, it will be overwritten.

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --vad_speech_fold_length # fold_length for speech data during VAD training (default="${vad_speech_fold_length}").
    --vad_text_fold_length   # fold_length for text data during VAD training (default="${vad_text_fold_length}").
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

num_inf=${num_inf:=${num_ref}}
# Preprocessor related
# For single speaker, text file path and name are text
ref_text_files_str="text "
ref_text_names_str="text "

# shellcheck disable=SC2206
ref_text_files=(${ref_text_files_str// / })
# shellcheck disable=SC2206
ref_text_names=(${ref_text_names_str// / })


# Set tag for naming of model directory
if [ -z "${vad_tag}" ]; then
    if [ -n "${vad_config}" ]; then
        vad_tag="$(basename "${vad_config}" .yaml)_${feats_type}"
    else
        vad_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${vad_args}" ]; then
        vad_tag+="$(echo "${vad_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${vad_stats_dir}" ]; then
    vad_stats_dir="${expdir}/vad_stats_${feats_type}"
fi
# The directory used for training commands
if [ -z "${vad_exp}" ]; then
    vad_exp="${expdir}/vad_${vad_tag}"
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
    inference_tag+="_vad_model_$(echo "${inference_vad_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

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
                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

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
                    "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 2: ${feats_type} extract: data/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # 2. Feature extract
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
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

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif  [ "${feats_type}" = extracted ]; then
            log "Stage 2: ${feats_type} extract: data/ -> ${data_feats}"
            # Assumming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                if [ ! -f data/"${dset}"/wav.scp ]; then
                    if [ ! -f data/"${dset}"/segments ]; then
                        <data/"${dset}"/feats.scp awk ' { print($1,"<DUMMY>") }' > data/"${dset}"/wav.scp
                    else
                        <data/"${dset}"/segments awk ' { print($2,"<DUMMY>") }' > data/"${dset}"/wav.scp
                    fi
                fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # Derive the the frame length and feature dimension
                _nj=$(min "${nj}" "$(<"${data_feats}${_suf}/${dset}/utt2spk" wc -l)")
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - | \
                    awk '{ print $2 }' | cut -d, -f2 > "${data_feats}${_suf}/${dset}/feats_dim"

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
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

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh \
                ${ref_text_files_str:+--utt_extra_files "${ref_text_files_str}"} \
                "${data_feats}/${dset}"
        done

    fi
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _vad_train_dir="${data_feats}/${train_set}"
        _vad_valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: VAD collect stats: train_set=${_vad_train_dir}, valid_set=${_vad_valid_dir}"

        _opts=
        if [ -n "${vad_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.vad_train --print_config --optim adam
            _opts+="--config ${vad_config} "
        fi

        _feats_type="$(<${_vad_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_vad_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${vad_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_vad_train_dir}/${_scp} wc -l)" "$(<${_vad_valid_dir}/${_scp} wc -l)")

        key_file="${_vad_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_vad_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${vad_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${vad_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${vad_stats_dir}/run.sh"; chmod +x "${vad_stats_dir}/run.sh"

        # 3. Submit jobs
        log "VAD collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        _opts+="--train_data_path_and_name_and_type ${_vad_train_dir}/${_scp},speech,${_type} "
        _opts+="--valid_data_path_and_name_and_type ${_vad_valid_dir}/${_scp},speech,${_type} "

	for i in ${!ref_text_files[@]}; do
            _opts+="--train_data_path_and_name_and_type ${_vad_train_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--valid_data_path_and_name_and_type ${_vad_valid_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
        done

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.vad_train \
                --collect_stats true \
                --use_preprocessor true \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${vad_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        if [ "${feats_normalize}" != global_mvn ]; then
            # Skip summerizaing stats if not using global MVN
            _opts+="--skip_sum_stats"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${vad_stats_dir}"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _vad_train_dir="${data_feats}/${train_set}"
        _vad_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: VAD Training: train_set=${_vad_train_dir}, valid_set=${_vad_valid_dir}"

        _opts=
        if [ -n "${vad_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.vad_train --print_config --optim adam
            _opts+="--config ${vad_config} "
        fi

        _feats_type="$(<${_vad_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((vad_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${vad_speech_fold_length}"
            _input_size="$(<${_vad_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${vad_stats_dir}/train/feats_stats.npz "
        fi

        _opts+="--train_data_path_and_name_and_type ${_vad_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_shape_file ${vad_stats_dir}/train/speech_shape "

        read -r -a aux_list <<< "$auxiliary_data_tags"
        if [ ${#aux_list[@]} != 0 ]; then
            _opts+="--allow_variable_data_keys True "
            for aux_dset in "${aux_list[@]}"; do
                 _opts+="--train_data_path_and_name_and_type ${_vad_train_dir}/${aux_dset},text,text "
            done
        fi
	    # shellcheck disable=SC2068
        for i in ${!ref_text_names[@]}; do
            _opts+="--fold_length ${vad_text_fold_length} "
            _opts+="--train_data_path_and_name_and_type ${_vad_train_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--train_shape_file ${vad_stats_dir}/train/${ref_text_names[$i]}_shape "
        done

	# shellcheck disable=SC2068
        for i in ${!ref_text_names[@]}; do
            _opts+="--valid_data_path_and_name_and_type ${_vad_valid_dir}/${ref_text_files[$i]},${ref_text_names[$i]},text "
            _opts+="--valid_shape_file ${vad_stats_dir}/valid/${ref_text_names[$i]}_shape "
        done

        log "Generate '${vad_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${vad_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${vad_exp}/run.sh"; chmod +x "${vad_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "VAD training started... log: '${vad_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${vad_exp})"
        else
            jobname="${vad_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${vad_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${vad_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.vad_train \
                --use_preprocessor true \
                --valid_data_path_and_name_and_type "${_vad_valid_dir}/${_scp},speech,${_type}" \
                --valid_shape_file "${vad_stats_dir}/valid/speech_shape" \
                --resume true \
                ${pretrained_model:+--init_param $pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --fold_length "${_fold_length}" \
                --output_dir "${vad_exp}" \
                ${_opts} ${vad_args}

    fi
else
    log "Skip the training stages"
fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Decoding: training_dir=${vad_exp}"

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
        log "Generate '${vad_exp}/${inference_tag}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${vad_exp}/${inference_tag}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${vad_exp}/${inference_tag}/run.sh"; chmod +x "${vad_exp}/${inference_tag}/run.sh"

        inference_bin_tag=""

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${vad_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
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

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/vad_inference.*.log'"
            rm -f "${_logdir}/*.log"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/vad_inference.JOB.log \
                ${python} -m espnet2.bin.vad_inference \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --vad_train_config "${vad_exp}"/config.yaml \
                    --vad_model_file "${vad_exp}"/"${inference_vad_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/vad_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            # shellcheck disable=SC2068
            for ref_txt in ${ref_text_files[@]}; do
                suffix=$(echo ${ref_txt} | sed 's/text//')
                if [ -f "${_logdir}/output.1/vad_result/segments" ]; then
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/vad_result/segments"
                    done | sort -k1 >"${_dir}/segments"
                fi
            done

        done
    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${vad_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"

            # shellcheck disable=SC2068
            for ref_txt in ${ref_text_files[@]}; do
                # Note(simpleoier): to get the suffix after text, e.g. "text_spk1" -> "_spk1"
                suffix=$(echo ${ref_txt} | sed 's/text//')

            done

            ${python} -m espnet2.bin.vad_scoring \
                --hyp_file "${_logdir}"/output.1/vad_result/segments \
                --ref_file "${_data}"/text \
                --output_file "${_dir}"/result.txt
        done

        # Show results in Markdown syntax
        scripts/utils/show_vad_result.sh "${vad_exp}" > "${vad_exp}"/RESULTS.md
        cat "${vad_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${vad_exp}/${vad_exp##*/}_${inference_vad_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Pack model: ${packed_model}"

        _opts=
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${vad_stats_dir}/train/feats_stats.npz "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack vad \
            --vad_train_config "${vad_exp}"/config.yaml \
            --vad_model_file "${vad_exp}"/"${inference_vad_model}" \
            ${_opts} \
            --option "${vad_exp}"/RESULTS.md \
            --option "${vad_exp}"/images \
            --outpath "${packed_model}"
    fi
fi

if ! "${skip_upload}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
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
        # /some/where/espnet/egs2/foo/vad1/ -> foo/vad1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/vad1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${vad_exp}"/description
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
<li><strong>Results</strong><pre><code>$(cat "${vad_exp}"/RESULTS.md)</code></pre></li>
<li><strong>VAD config</strong><pre><code>$(cat "${vad_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}" \
            --description_file "${vad_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stage"
fi

if ! "${skip_upload_hf}"; then
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        [ -z "${hf_repo}" ] && \
            log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
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
        # /some/where/espnet/egs2/foo/vad1/ -> foo/vad1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/vad1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # copy files in ${dir_repo}
        unzip -o ${packed_model} -d ${dir_repo}
        # Generate description file
        # shellcheck disable=SC2034
        hf_task=voice-activity-detection
        # shellcheck disable=SC2034
        espnet_task=VAD
        # shellcheck disable=SC2034
        task_exp=${vad_exp}
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
