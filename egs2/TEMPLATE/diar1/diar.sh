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
hop_length=128       # Hop length in sample number
min_wav_duration=0.1 # Minimum duration in second

# diar model related
diar_tag=    # Suffix to the result dir for diar model training.
diar_config= # Config for diar model training.
diar_args=   # Arguments for diar model training, e.g., "--max_epoch 10".
             # Note that it will overwrite args in diar config.
feats_normalize=global_mvn # Normalizaton layer type.
num_spk=2    # Number of speakers in the input audio

# diar related
inference_config= # Config for diar model inference
inference_model=valid.acc.best.pth
inference_tag=    # Suffix to the inference dir for diar model inference
download_model=   # Download a model from Model Zoo and use it for diarization.

# Upload model related
hf_repo=

# scoring related
collar=0         # collar for der scoring
frame_shift=128  # frame shift to convert frame-level label into real time
                 # this should be aligned with frontend feature extraction

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
    --hop_length       # Hop length in sample number (default="${hop_length}")
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").


    # Diarization model related
    --diar_tag        # Suffix to the result dir for diarization model training (default="${diar_tag}").
    --diar_config     # Config for diarization model training (default="${diar_config}").
    --diar_args       # Arguments for diarization model training, e.g., "--max_epoch 10" (default="${diar_args}").
                      # Note that it will overwrite args in diar config.
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").
    --num_spk         # Number of speakers in the input audio (default="${num_spk}")

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
diar_stats_dir="${expdir}/diar_stats_${fs}"
# The directory used for training commands
diar_exp="${expdir}/diar_${diar_tag}"

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
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}"  \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
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

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            # diarization typically accept long recordings, so does not has
            # max length requirements
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" \
                    '{ if ($2 > min_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"

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
        log "Stage 4: Diarization collect stats: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

        _opts=
        if [ -n "${diar_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.diar_train --print_config --optim adam
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
            _opts+="--frontend_conf fs=${fs} "
            _opts+="--frontend_conf hop_length=${hop_length} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        _opts+="--num_spk ${num_spk} "

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
        log "Diarization collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.diar_train \
                --collect_stats true \
                --use_preprocessor true \
                --train_data_path_and_name_and_type "${_diar_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_diar_train_dir}/espnet_rttm,spk_labels,rttm" \
                --valid_data_path_and_name_and_type "${_diar_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_diar_valid_dir}/espnet_rttm,spk_labels,rttm" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${diar_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

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
        log "Stage 5: Diarization Training: train_set=${_diar_train_dir}, valid_set=${_diar_valid_dir}"

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
            _opts+="--frontend_conf fs=${fs} "
            _opts+="--frontend_conf hop_length=${hop_length} "
        else
            echo "does not support other feats_type (i.e., ${_feats_type}) now"
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${diar_stats_dir}/train/feats_stats.npz "
        fi

        _opts+="--num_spk ${num_spk} "

        _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_diar_train_dir}/espnet_rttm,spk_labels,rttm "
        _opts+="--train_shape_file ${diar_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${diar_stats_dir}/train/spk_labels_shape "

        _opts+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/${_scp},speech,${_type} "
        _opts+="--valid_data_path_and_name_and_type ${_diar_valid_dir}/espnet_rttm,spk_labels,rttm "
        _opts+="--valid_shape_file ${diar_stats_dir}/valid/speech_shape "
        _opts+="--valid_shape_file ${diar_stats_dir}/valid/spk_labels_shape "

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
            ${python} -m espnet2.bin.diar_train \
                --use_preprocessor true \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${diar_speech_fold_length}" \
                --output_dir "${diar_exp}" \
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
        log "Stage 6: Diarize Speaker: training_dir=${diar_exp}"

        if ${gpu_inference}; then
            _cmd=${cuda_cmd}
            _ngpu=1
        else
            _cmd=${decode_cmd}
            _ngpu=0
        fi

        log "Generate '${diar_exp}/run_diarize.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${diar_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${diar_exp}/run_diarize.sh"; chmod +x "${diar_exp}/run_diarize.sh"
        _opts=

        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${diar_exp}/diarized_${dset}"
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
            log "Diarization started... log: '${_logdir}/diar_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/diar_inference.JOB.log \
                ${python} -m espnet2.bin.diar_inference \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${diar_exp}"/config.yaml \
                    --model_file "${diar_exp}"/"${inference_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} || { cat $(grep -l -i error "${_logdir}"/diar_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/diarize.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/diarize.scp"

        done
    fi

    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Scoring"
        _cmd=${decode_cmd}

        for dset in "${valid_set}" ${test_sets}; do
            _data="${data_feats}/${dset}"
            _inf_dir="${diar_exp}/diarized_${dset}"
            _dir="${diar_exp}/diarized_${dset}/scoring"
            mkdir -p "${_dir}"

            scripts/utils/score_der.sh ${_dir} ${_inf_dir}/diarize.scp ${_data}/rttm \
                --collar ${collar} --fs ${fs} --frame_shift ${frame_shift}
        done

        # Show results in Markdown syntax
        scripts/utils/show_diar_result.sh "${diar_exp}" > "${diar_exp}"/RESULTS.md
        cat "${diar_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${diar_exp}/${diar_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Pack model: ${packed_model}"

        ${python} -m espnet2.bin.pack diar \
            --train_config "${diar_exp}"/config.yaml \
            --model_file "${diar_exp}"/"${inference_model}" \
            --option "${diar_exp}"/RESULTS.md \
            --option "${diar_stats_dir}"/train/feats_stats.npz  \
            --option "${diar_exp}"/images \
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
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
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
