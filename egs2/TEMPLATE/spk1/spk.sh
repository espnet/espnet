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
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_stages=          # Spicify the stage to be skipped
skip_data_prep=false  # Skip data preparation stages.
skip_train=false      # Skip training stages.
skip_eval=false       # Skip decoding and evaluation stages.
skip_upload_hf=true   # Skip uploading to hugging face stages.

eval_valid_set=false  # Run decoding for the validation set
ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
gpu_inference=false   # Whether to perform gpu decoding.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.
python=python3        # Specify python to execute espnet commands.
fold_length=120000    # fold_length for speech data during enhancement training.

# Data preparation related
local_data_opts= # The options given to local/data.sh

# Speed perturbation related
speed_perturb_factors="0.9 1.0 1.1" # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw      # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.
min_wav_duration=1.0  # Minimum duration in second.
max_wav_duration=60.  # Maximum duration in second.

# Speaker model related
spk_exp=              # Specify the directory path for spk experiment.
spk_tag=              # Suffix to the result dir for spk model training.
spk_config=           # Config for the spk model training.
spk_args=             # Arguments for spk model training.
pretrained_model=     # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch

# Inference related
inference_config=conf/decode.yaml   # Inference configuration
inference_model=valid.eer.best.pth  # Inference model weight file
score_norm=false      # Apply score normalization in inference.
qmf_func=false        # Apply quality measurement based calibration in inference.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=        # Name of training set.
valid_set=        # Name of validation set used for monitoring/tuning network training.
test_sets=        # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
lang=multilingual # The language type of corpus.


# Upload model related
hf_repo=

help_message=$(cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    stage=1               # Processes starts from the specified stage.
    stop_stage=10000      # Processes is stopped at the specified stage.
    skip_stages=          # Spicify the stage to be skipped
    skip_data_prep=false  # Skip data preparation stages.
    skip_train=false      # Skip training stages.
    skip_eval=false       # Skip decoding and evaluation stages.
    skip_upload_hf        # Skip packing and uploading stages (default="${skip_upload_hf}").

    eval_valid_set=false  # Run decoding for the validation set
    ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
    num_nodes=1           # The number of nodes.
    nj=32                 # The number of parallel jobs.
    gpu_inference=false   # Whether to perform gpu decoding.
    dumpdir=dump          # Directory to dump features.
    expdir=exp            # Directory to save experiments.
    python=python3        # Specify python to execute espnet commands.
    fold_length=80000     # fold_length for speech data during enhancement training

    # Speed perturbation related
    speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

    # Feature extraction related
    feats_type=raw       # Feature type (raw, raw_copy, fbank_pitch, or extracted).
    audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
    multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
    multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
    fs=16k               # Sampling rate.
    min_wav_duration=1.0  # Minimum duration in second.
    max_wav_duration=60.  # Maximum duration in second.

    # Speaker model related
    spk_exp=              # Specify the directory path for spk experiment.
    spk_tag=              # Suffix to the result dir for spk model training.
    spk_config=           # Config for the spk model training.
    spk_args=             # Arguments for spk model training.
    pretrained_model=     # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch= # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").

    # Inference related
    inference_config=     # Inference configuration file
    inference_model=      # Inference model weight file
    score_norm=false      # Apply score normalization in inference.
    qmf_func=false        # Apply quality measurement based calibration in inference.

    # [Task dependent] Set the datadir name created by local/data.sh
    train_set=        # Name of training set.
    valid_set=        # Name of validation set used for monitoring/tuning network training.
    test_sets=        # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
    lang=multilingual # The language type of corpus.

    # Upload model related
    hf_repo=          # The huggingface repository directory

EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0  ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
        exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = raw  ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = raw_copy  ]; then
    # raw_copy is as same as raw except for skipping the format_wav stage
    data_feats=${dumpdir}/raw_copy
elif [ "${feats_type}" = fbank  ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" = extracted  ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for speaker recognition process
utt_extra_files="utt2category"

# Set tag for naming of model directory
if [ -z "${spk_tag}" ]; then
    if [ -n "${spk_config}" ]; then
        spk_tag="$(basename "${spk_config}" .yaml)_${feats_type}"
    else
        spk_tag="train_${feats_type}"
    fi
fi

# Set directory used for training commands
spk_stats_dir="${expdir}/spk_stats_${fs}"
if [ -z "${spk_exp}"  ]; then
    spk_exp="${expdir}/spk_${spk_tag}"
fi

# Determine which stages to skip
if "${skip_data_prep}"; then
    skip_stages+="1 2 "
fi

if "${skip_upload_hf}"; then
    skip_stages+="9 10 "
fi

skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
    log "Stage 1 FIN."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
        log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
        # For example, when speed_perturb_factors="0.9 1.0 1.1", the number of unique speakers will be increased by three times
        _scp_list="wav.scp "

        for factor in ${speed_perturb_factors}; do
            if ${python} -c "assert ${factor} != 1.0" 2>/dev/null; then
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
    spk_stats_dir="${spk_stats_dir}_sp"
    spk_exp="${spk_exp}_sp"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

    if "${skip_train}"; then
        if "${eval_valid_set}"; then
            _dsets="${valid_set} ${test_sets}"
        else
            _dsets="${test_sets}"
        fi
    else
        _dsets="${valid_set} ${test_sets}"
    fi

    if [ "${feats_type}" = raw ]; then
        if [ "${skip_train}" = false ]; then
            utils/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"

            # copy extra files that are not covered by copy_data_dir.sh
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${train_set}/wav.scp" "${data_feats}/${train_set}"

            echo "${feats_type}" > "${data_feats}/${train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${train_set}/audio_format"
            fi
        fi

        # Calculate EER for valid/test since speaker verification is an open set problem
        # Train can be either multi-column data or not, but valid/test always require multi-column trial
        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"

            # copy extra files that are not covered by copy_data_dir.sh
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            cp data/${dset}/trial_label "${data_feats}/${dset}"

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                --out_filename trial.scp \
                "data/${dset}/trial.scp" "${data_feats}/${dset}"
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                --out_filename trial2.scp \
                "data/${dset}/trial2.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"

        done
    elif [ "${feats_type}" = raw_copy ]; then
        if [ "${skip_train}" = false ]; then
            utils/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"
            # category2utt will be used bydata sampler
            cp data/"${train_set}/spk2utt" "${data_feats}/${train_set}/category2utt"
            for x in music noise speech; do
                cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
            done
            cp data/rirs.scp ${data_feats}/rirs.scp

            echo "${feats_type}" > "${data_feats}/${train_set}/feats_type"
            if "${multi_columns_output_wav_scp}"; then
                echo "multi_${audio_format}" > "${data_feats}/${train_set}/audio_format"
            else
                echo "${audio_format}" > "${data_feats}/${train_set}/audio_format"
            fi
        fi

        # Calculate EER for valid/test since speaker verification is an open set problem
        # Train can be either multi-column data or not, but valid/test always require multi-column trial
        for dset in ${_dsets}; do
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"
            cp data/${dset}/trial_label "${data_feats}/${dset}"
            cp data/${dset}/trial.scp "${data_feats}/${dset}"
            cp data/${dset}/trial2.scp "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "multi_${audio_format}" > "${data_feats}/${dset}/audio_format"

        done

        for f in ${utt_extra_files}; do
            [ -f data/${dset}/${f} ] && cp data/${dset}/${f} ${data_feats}/${dset}/${f}
        done
    else
        log "${feats_type} is not supported yet."
        exit 1
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Collect stats"
    _spk_train_dir="${data_feats}/${train_set}"
    _spk_valid_dir="${data_feats}/${valid_set}"

    if [ -n "${spk_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${spk_config} "
    fi

    if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    else
        # sound supports "wav", "flac", etc.
        _type=sound
    fi

    # 1. Split key file
    _logdir="${spk_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    _nj=$(min "${nj}" "$(<${_spk_train_dir}/wav.scp wc -l)" "$(<${_spk_valid_dir}/trial.scp wc -l)")

    key_file="${_spk_train_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_spk_valid_dir}/trial.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${spk_stats_dir}/run.sh'. You can resume the process from stage 3 using this script"
    mkdir -p "${spk_stats_dir}"; echo "${run_args} -- stage3 \"\$@\"; exit \$?" > "${spk_stats_dir}/run.sh"; chmod +x "${spk_stats_dir}/run.sh"

    # 3. Submit jobs
    log "Speaker collect-stats started... log: '${_logdir}/stats.*.log'"

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.spk_train \
            --collect_stats true \
            --use_preprocessor false \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,${_type} \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial.scp,speech,${_type} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --spk2utt ${_spk_train_dir}/spk2utt \
            --spk_num $(wc -l ${_spk_train_dir}/spk2utt | cut -f1 -d" ") \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${spk_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1;  }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --skip_sum_stats --output_dir "${spk_stats_dir}"

    cp ${spk_stats_dir}/valid/speech_shape ${spk_stats_dir}/valid/speech_shape2
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Train."

    _spk_train_dir="${data_feats}/${train_set}"
    _spk_valid_dir="${data_feats}/${valid_set}"
    _opts=
    if [ -n "${spk_config}"  ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.spk_train --print_config --optim adam
        _opts+="--config ${spk_config} "
    fi

    log "Spk training started... log: '${spk_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${spk_exp})"
    else
        jobname="${spk_exp}/train.log"
    fi

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${spk_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${spk_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_train \
            --use_preprocessor true \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --output_dir ${spk_exp} \
            --train_data_path_and_name_and_type ${_spk_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_spk_train_dir}/utt2spk,spk_labels,text \
            --train_shape_file ${spk_stats_dir}/train/speech_shape \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial.scp,speech,sound \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial2.scp,speech2,sound \
            --valid_data_path_and_name_and_type ${_spk_valid_dir}/trial_label,spk_labels,text \
            --spk2utt ${_spk_train_dir}/spk2utt \
            --spk_num $(wc -l ${_spk_train_dir}/spk2utt | cut -f1 -d" ") \
            --fold_length ${fold_length} \
            --valid_shape_file ${spk_stats_dir}/valid/speech_shape \
            --output_dir "${spk_exp}" \
            ${_opts} ${spk_args}
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Speaker embedding extraction."

    infer_exp="${spk_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}
    log "Extracting speaker embeddings for inference... log: '${infer_exp}/spk_embed_extraction.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${infer_exp})"
    else
        jobname="${infer_exp}/spk_embed_extraction.log"
    fi

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${infer_exp}/spk_embed_extraction_test.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${spk_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.spk_embed_extract \
            --use_preprocessor true \
            --output_dir ${infer_exp} \
            --data_path_and_name_and_type ${_inference_dir}/trial.scp,speech,sound \
            --data_path_and_name_and_type ${_inference_dir}/trial2.scp,speech2,sound \
            --data_path_and_name_and_type ${_inference_dir}/trial_label,spk_labels,text \
            --shape_file ${spk_stats_dir}/valid/speech_shape \
            --fold_length ${fold_length} \
            --config ${inference_config} \
            --spk_train_config "${spk_exp}/config.yaml" \
            --spk_model_file "${spk_exp}"/${inference_model} \
            ${spk_args}

    # extract embeddings for cohort set
    if [ "$score_norm" = true  ] || [ "$qmf_func" = true  ]; then
        _spk_train_dir="${data_feats}/${train_set}"
        if [ ! -e "${_spk_train_dir}/cohort.scp"  ]; then
            ${python} pyscripts/utils/generate_cohort_list.py ${_spk_train_dir}/spk2utt ${_spk_train_dir}/wav.scp ${_spk_train_dir} ${inference_config} ${fs}
        fi
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log ${infer_exp}/spk_embed_extraction_cohort.log \
            --ngpu ${ngpu} \
            --num_nodes ${num_nodes} \
            --init_file_prefix ${spk_exp}/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.spk_embed_extract \
                --use_preprocessor true \
                --output_dir ${infer_exp} \
                --data_path_and_name_and_type ${_spk_train_dir}/cohort.scp,speech,sound \
                --data_path_and_name_and_type ${_spk_train_dir}/cohort2.scp,speech2,sound \
                --data_path_and_name_and_type ${_spk_train_dir}/cohort_label,spk_labels,text \
                --shape_file ${_spk_train_dir}/cohort_speech_shape \
                --fold_length ${fold_length} \
                --config ${inference_config} \
                --spk_train_config "${spk_exp}/config.yaml" \
                --spk_model_file "${spk_exp}"/${inference_model} \
                --average_embd "true" \
                ${spk_args}
    fi

    # extract embeddings for qmf train set
    if "$qmf_func"; then
        _spk_train_dir="${data_feats}/${train_set}"
        if [ ! -e "${_spk_train_dir}/qmf_train.scp"  ]; then
            ${python} pyscripts/utils/generate_qmf_train_list.py ${_spk_train_dir}/spk2utt ${_spk_train_dir}/wav.scp ${_spk_train_dir} ${inference_config} ${_spk_train_dir}/utt2spk ${_spk_train_dir}/cohort_label ${fs}
            mkdir ${infer_exp}/qmf
        fi
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log ${infer_exp}/spk_embed_extraction_qmf_train.log \
            --ngpu ${ngpu} \
            --num_nodes ${num_nodes} \
            --init_file_prefix ${spk_exp}/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.spk_embed_extract \
                --use_preprocessor true \
                --output_dir ${infer_exp}/qmf \
                --data_path_and_name_and_type ${_spk_train_dir}/qmf_train.scp,speech,sound \
                --data_path_and_name_and_type ${_spk_train_dir}/qmf_train2.scp,speech2,sound \
                --data_path_and_name_and_type ${_spk_train_dir}/qmf_train_label,spk_labels,text \
                --shape_file ${_spk_train_dir}/qmf_train_speech_shape \
                --fold_length ${fold_length} \
                --config ${inference_config} \
                --spk_train_config "${spk_exp}/config.yaml" \
                --spk_model_file "${spk_exp}"/${inference_model} \
                ${spk_args}
    fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Score calculation and post-processing."

    infer_exp="${spk_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}
    _spk_train_dir="${data_feats}/${train_set}"

    log "Stage 7-a: get scores for the test set."
    ${python} pyscripts/utils/spk_calculate_scores_from_embeddings.py ${infer_exp}/${test_sets}_embeddings.npz ${_inference_dir}/trial_label ${infer_exp}/${test_sets}_raw_trial_scores
    scorefile_cur=${infer_exp}/${test_sets}_raw_trial_scores

    if "$score_norm"; then
        log "Stage 7-b: apply score normalization."
        ${python} pyscripts/utils/spk_apply_score_norm.py ${scorefile_cur} ${infer_exp}/${test_sets}_embeddings.npz ${infer_exp}/${train_set}_embeddings.npz ${_spk_train_dir}/utt2spk ${infer_exp}/${test_sets}_scorenormed_scores ${inference_config} ${ngpu}
        scorefile_cur=${infer_exp}/${test_sets}_scorenormed_scores
    fi

    if "$qmf_func"; then
        log "Stage 7-c: apply QMF calibration."
        log "get raw scores for the qmf train set."
        ${python} pyscripts/utils/spk_calculate_scores_from_embeddings.py ${infer_exp}/qmf/${train_set}_embeddings.npz ${_spk_train_dir}/qmf_train_label ${infer_exp}/qmf/${train_set}_raw_trial_scores

        if "$score_norm"; then
            log "normalize qmf train set scores."
            ${python} pyscripts/utils/spk_apply_score_norm.py ${infer_exp}/qmf/${train_set}_raw_trial_scores ${infer_exp}/qmf/${train_set}_embeddings.npz ${infer_exp}/${train_set}_embeddings.npz ${_spk_train_dir}/utt2spk ${infer_exp}/qmf/${train_set}_scorenormed_scores ${inference_config} ${ngpu}
            qmf_train_scores=${infer_exp}/qmf/${train_set}_scorenormed_scores
            test_scores=${infer_exp}/${test_sets}_scorenormed_scores
        else
            qmf_train_scores=${infer_exp}/qmf/${train_set}_raw_trial_scores
            test_scores=${infer_exp}/${test_sets}_raw_trial_scores
        fi

        log "Apply qmf function."
        ${python} pyscripts/utils/spk_apply_qmf_func.py ${_spk_train_dir}/qmf_train.scp ${_spk_train_dir}/qmf_train2.scp ${qmf_train_scores} ${infer_exp}/qmf/${train_set}_embeddings.npz ${_inference_dir}/trial.scp ${_inference_dir}/trial2.scp ${test_scores} ${infer_exp}/${test_sets}_embeddings.npz ${infer_exp}/qmf/${test_sets}_qmf_scores
    fi

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Calculate metrics."
    infer_exp="${spk_exp}/inference"
    _inference_dir=${data_feats}/${test_sets}

    if "$score_norm"; then
        if "$qmf_func"; then
            score_dir=${infer_exp}/qmf/${test_sets}_qmf_scores
        else
            score_dir=${infer_exp}/${test_sets}_scorenormed_scores
        fi
    else
        if "$qmf_func"; then
            score_dir=${infer_exp}/qmf/${test_sets}_qmf_scores
        else
            score_dir=${infer_exp}/${test_sets}_raw_trial_scores
        fi
    fi

    log "calculate score with ${score_dir}"
    ${python} pyscripts/utils/calculate_eer_mindcf.py ${score_dir} ${infer_exp}/${test_sets}_metrics

    # Show results in Markdown syntax
    ${python} scripts/utils/show_spk_result.py "${infer_exp}/${test_sets}_metrics" "${spk_exp}"/RESULTS.md $(echo ${spk_config} | cut -d'.' -f1)
    cat "${spk_exp}"/RESULTS.md
fi

packed_model="${spk_exp}/${spk_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    log "Stage 9: Pack model: ${packed_model}"

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack spk \
        --train_config "${spk_exp}"/config.yaml \
        --model_file "${spk_exp}"/"${inference_model}" \
        --option "${spk_exp}"/RESULTS.md \
        --option "${spk_exp}"/images \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Upload model to HuggingFace: ${hf_repo}"
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1

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
    # /some/where/espnet/egs2/foo/spk1/ -> foo/spk1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=speaker-recognition
    # shellcheck disable=SC2034
    espnet_task=SPK
    # shellcheck disable=SC2034
    task_exp=${spk_exp}
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

log "Successfully finished. [elapsed=${SECONDS}s]"
