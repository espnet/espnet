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
skip_packing=true     # Skip the packing stage.
skip_upload_hf=true   # Skip uploading to huggingface stage.

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
speed_perturb_factors="" # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw      # Feature type (raw, raw_copy, fbank_pitch, or extracted).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
multi_columns_input_wav_scp=false  # Enable multi columns mode for input wav.scp for format_wav_scp.py
multi_columns_output_wav_scp=false # Enable multi columns mode for output wav.scp for format_wav_scp.py
fs=16k               # Sampling rate.
min_wav_duration=1.0  # Minimum duration in second.
max_wav_duration=60.  # Maximum duration in second.

# Language identification model related
lid_exp=              # Specify the directory path for lid experiment.
lid_tag=              # Suffix to the result dir for lid model training.
lid_config=           # Config for the lid model training.
lid_args=             # Arguments for lid model training.
pretrained_model=     # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch

# Inference related
inference_model=valid.loss.best.pth  # Inference model weight file
inference_batch_size=1
extract_embd=false             # Whether to extract embeddings per utt
checkpoint_interval=1000       # Save checkpoint every N utterances during inference
max_utt_per_lang_for_tsne=1000 # Maximum number of utterances per language for t-SNE visualization
perplexity=5                   # The perplexity for t-SNE
max_iter=1000                  # The maximum number of iterations for t-SNE

# [Task dependent] Set the datadir name created by local/data.sh
train_set=        # Name of training set.
valid_set=        # Name of validation set used for monitoring/tuning network training.
test_sets=        # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
tsne_set=         # Name of set for t-SNE visualization, typically the train set


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
    skip_packing=true     # Skip the packing stage.
    skip_upload_hf=true   # Skip uploading to huggingface stage.

    eval_valid_set=false  # Run decoding for the validation set
    ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
    num_nodes=1           # The number of nodes.
    nj=32                 # The number of parallel jobs.
    gpu_inference=false   # Whether to perform gpu decoding.
    dumpdir=dump          # Directory to dump features.
    expdir=exp            # Directory to save experiments.
    python=python3        # Specify python to execute espnet commands.
    fold_length=120000     # fold_length for speech data during enhancement training

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

    # Language identification model related
    lid_exp=              # Specify the directory path for lid experiment.
    lid_tag=              # Suffix to the result dir for lid model training.
    lid_config=           # Config for the lid model training.
    lid_args=             # Arguments for lid model training.
    pretrained_model=     # Pretrained model to load (default="${pretrained_model}").
    ignore_init_mismatch= # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").

    # Inference related
    inference_model=      # Inference model weight file
    inference_batch_size= # Inference batch size
    extract_embd=         # Whether to extract embeddings or not
    checkpoint_interval=  # Save checkpoint every N utterances during inference
    max_utt_per_lang_for_tsne=1000 # Maximum number of utterances per language for t-SNE visualization
    perplexity=5                   # The perplexity for t-SNE
    max_iter=1000                  # The maximum number of iterations for t-SNE

    # [Task dependent] Set the datadir name created by local/data.sh
    train_set=        # Name of training set.
    valid_set=        # Name of validation set used for monitoring/tuning network training.
    test_sets=        # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

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

# Extra files for language identification process
utt_extra_files="utt2category"

# Set tag for naming of model directory
if [ -z "${lid_tag}" ]; then
    if [ -n "${lid_config}" ]; then
        lid_tag="$(basename "${lid_config}" .yaml)_${feats_type}"
    else
        lid_tag="train_${feats_type}"
    fi
fi

# Set directory used for training commands
lid_stats_dir="${expdir}/lid_stats_${fs}"
if [ -z "${lid_exp}" ]; then
    lid_exp="${expdir}/lid_${lid_tag}"
fi

# Determine which stages to skip
if "${skip_data_prep}"; then
    skip_stages+="1 2 "
fi

if "${skip_packing}"; then
    skip_stages+="9 "
fi
if "${skip_upload_hf}"; then
    skip_stages+="10 "
fi

test_sets_ood="" # out-of-domain test sets
test_sets_id=""  # in-domain test sets
test_sets_all=""
train_name=$(echo "${train_set}" | cut -d'_' -f2)

# Check if the test set is in-domain or out-of-domain
for test_set in ${test_sets}; do
    test_name=$(echo "${test_set}" | cut -d'_' -f2)
    if [ "${train_name}" != "${test_name}" ]; then
        test_sets_ood+="${test_set} "
        test_sets_all+="${test_set}_cross_${train_set} "
    else
        test_sets_id+="${test_set} "
        test_sets_all+="${test_set} "
    fi
done

skip_stages=$(echo "${skip_stages}" | tr ' ' '\n' | sort -nu | tr '\n' ' ')
log "Skipped stages: ${skip_stages}"


if [ ${stage} -le 1  ] && [ ${stop_stage} -ge 1  ] && ! [[ " ${skip_stages} " =~ [[:space:]]1[[:space:]]  ]]; then
    log "Stage 1: Data preparation for train and evaluation."
    # [Task dependent] Need to create data.sh for new corpus
    # Please prepare utt2lang, lang2utt, wav.scp, segments (optional)
    # for train/dev/test sets.
    local/data.sh ${local_data_opts}
    log "Stage 1 Complete."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! [[ " ${skip_stages} " =~ [[:space:]]2[[:space:]] ]]; then
    if [ -n "${speed_perturb_factors}" ]; then
        log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

        _scp_list="wav.scp "

        # Temporary move to use perturb_lid_data_dir_speed.sh
        mv "data/${train_set}/utt2lang" "data/${train_set}/utt2spk"
        mv "data/${train_set}/lang2utt" "data/${train_set}/spk2utt"

        for factor in ${speed_perturb_factors}; do
            if [ "${factor}" != "1.0" ] && [ "${factor}" != "1" ]; then
                local/perturb_lid_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
                _dirs+="data/${train_set}_sp${factor} "
            else
                # If speed factor is 1, same as the original
                _dirs+="data/${train_set} "
            fi
        done
        utils/combine_data.sh --extra-files "${_scp_list}" "data/${train_set}_sp" ${_dirs}

        # Restore the original files
        for dir in ${_dirs}; do
            mv "${dir}/utt2spk" "${dir}/utt2lang"
            mv "${dir}/spk2utt" "${dir}/lang2utt"
        done
        mv "data/${train_set}_sp/utt2spk" "data/${train_set}_sp/utt2lang"
        mv "data/${train_set}_sp/spk2utt" "data/${train_set}_sp/lang2utt"
    else
        log "Skip stage 2: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
    lid_stats_dir="${lid_stats_dir}_sp"
    lid_exp="${lid_exp}_sp"
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
            local/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"

            # Copy extra files that are not covered by copy_data_dir.sh
            # category2utt will be used by data sampler
            cp data/"${train_set}/lang2utt" "${data_feats}/${train_set}/category2utt"

            for x in music noise speech; do
                if [ -f data/musan_${x}.scp ]; then
                    cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
                fi
            done
            if [ -f data/rirs.scp ]; then
                cp data/rirs.scp ${data_feats}/rirs.scp
            fi

            _opts=
            if [ -e data/"${train_set}"/segments ]; then
                # "segments" is used to split audio files listed in "wav.scp" into utterances.
                # The "segments" file format:
                #   <segment_id> <record_id> <start_time> <end_time>
                # Example:
                #   call-861225-A-0050-0065 call-861225-A 5.0 6.5
                # - <segment_id>: Unique ID for the utterance segment
                # - <record_id>: Corresponding recording ID from wav.scp
                # - <start_time> and <end_time>: Start and end times in seconds
                _opts+="--segments data/${train_set}/segments "
            fi

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
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

        for dset in ${_dsets}; do
            local/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"

            cp data/"${dset}/lang2utt" "${data_feats}/${dset}/category2utt"

            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi

            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                --multi-columns-input "${multi_columns_input_wav_scp}" \
                --multi-columns-output "${multi_columns_output_wav_scp}" \
                "data/${dset}/wav.scp" "${data_feats}/${dset}"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "${audio_format}" > "${data_feats}/${dset}/audio_format"

        done
    elif [ "${feats_type}" = raw_copy ]; then
        if [ "${skip_train}" = false ]; then
            local/copy_data_dir.sh --validate_opts --non-print data/"${train_set}" "${data_feats}/${train_set}"

            cp data/"${train_set}/lang2utt" "${data_feats}/${train_set}/category2utt"

            for x in music noise speech; do
                if [ -f data/musan_${x}.scp ]; then
                    cp data/musan_${x}.scp ${data_feats}/musan_${x}.scp
                fi
            done
            if [ -f data/rirs.scp ]; then
                cp data/rirs.scp ${data_feats}/rirs.scp
            fi

            echo "${feats_type}" > "${data_feats}/${train_set}/feats_type"
            echo "${audio_format}" > "${data_feats}/${train_set}/audio_format"
        fi

        for dset in ${_dsets}; do
            local/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}/${dset}"

            cp data/"${dset}/lang2utt" "${data_feats}/${dset}/category2utt"

            echo "${feats_type}" > "${data_feats}/${dset}/feats_type"
            echo "${audio_format}" > "${data_feats}/${dset}/audio_format"

        done

    else
        log "${feats_type} is not supported yet."
        exit 1
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Collect stats"
    _lid_train_dir="${data_feats}/${train_set}"
    _lid_valid_dir="${data_feats}/${valid_set}"

    if [ -n "${lid_config}"  ]; then
        _opts+="--config ${lid_config} "
    fi

    if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    else
        # sound supports "wav", "flac", etc.
        _type=sound
    fi

    # 1. Split key file
    _logdir="${lid_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    _nj=$(min "${nj}" "$(<${_lid_train_dir}/wav.scp wc -l)" "$(<${_lid_valid_dir}/wav.scp wc -l)")

    key_file="${_lid_train_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_lid_valid_dir}/wav.scp"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Generate run.sh
    log "Generate '${lid_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
    mkdir -p "${lid_stats_dir}"; echo "${run_args} --stage 4 \"\$@\"; exit \$?" > "${lid_stats_dir}/run.sh"; chmod +x "${lid_stats_dir}/run.sh"

    # 3. Submit jobs
    log "Language identification collect-stats started... log: '${_logdir}/stats.*.log'"

    # shellcheck disable=SC2046,SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.lid_train \
            --collect_stats true \
            --use_preprocessor false \
            --train_data_path_and_name_and_type ${_lid_train_dir}/wav.scp,speech,${_type} \
            --valid_data_path_and_name_and_type ${_lid_valid_dir}/wav.scp,speech,${_type} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --lang2utt ${_lid_train_dir}/lang2utt \
            --lang_num $(wc -l ${_lid_train_dir}/lang2utt | cut -f1 -d" ") \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${lid_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1;  }

    # 4. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --skip_sum_stats --output_dir "${lid_stats_dir}"

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Train."

    _lid_train_dir="${data_feats}/${train_set}"
    _lid_valid_dir="${data_feats}/${valid_set}"
    _opts=
    if [ -n "${lid_config}"  ]; then
        _opts+="--config ${lid_config} "
    fi

    log "LID training started... log: '${lid_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${lid_exp})"
    else
        jobname="${lid_exp}/train.log"
    fi

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${lid_exp}/train.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${lid_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.lid_train \
            --use_preprocessor true \
            --resume true \
            ${pretrained_model:+--init_param $pretrained_model} \
            --ignore_init_mismatch ${ignore_init_mismatch} \
            --output_dir ${lid_exp} \
            --train_data_path_and_name_and_type ${_lid_train_dir}/wav.scp,speech,sound \
            --train_data_path_and_name_and_type ${_lid_train_dir}/utt2lang,lid_labels,text \
            --train_shape_file ${lid_stats_dir}/train/speech_shape \
            --valid_data_path_and_name_and_type ${_lid_valid_dir}/wav.scp,speech,sound \
            --valid_data_path_and_name_and_type ${_lid_valid_dir}/utt2lang,lid_labels,text \
            --lang2utt ${_lid_train_dir}/lang2utt \
            --lang_num $(wc -l ${_lid_train_dir}/lang2utt | cut -f1 -d" ") \
            --fold_length ${fold_length} \
            --valid_shape_file ${lid_stats_dir}/valid/speech_shape \
            ${_opts} ${lid_args}
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Language identification and embedding extraction on test sets."

    log "Prepare out-of-domain test sets."

    # Prepare out-of-domain test sets
    if [ -n "${test_sets_ood}" ]; then
        log "Out-of-domain test sets: ${test_sets_ood}"
        ./local/prepare_ood_test.sh \
            --dump_dir ${data_feats} \
            --train_set ${train_set} \
            --test_sets "${test_sets_ood}"
    fi

    inference_model_name="${inference_model%.pth}"
    log "Inference model name: ${inference_model_name}"
    log "Test sets after being processed: ${test_sets}"
    for test_set in ${test_sets_all}; do
        infer_exp="${lid_exp}/inference/${inference_model_name}/${test_set}"
        if [ -f "${infer_exp}/${test_set}_lids" ] && [ -f "${infer_exp}/${test_set}_lang_to_embds.npz" ]; then
            log "Skip inference for ${test_set} since it has already been done."
            continue
        fi
        _inference_dir=${data_feats}/${test_set}

        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${infer_exp})"
        else
            jobname="${infer_exp}/inference_embd_lid.log"
        fi

        log "Extracting language embeddings and ids... log: '${infer_exp}/inference_embd_lid.log'"
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log ${infer_exp}/inference_embd_lid.log \
            --ngpu ${ngpu} \
            --num_nodes ${num_nodes} \
            --init_file_prefix ${lid_exp}/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.lid_inference \
                --output_dir ${infer_exp} \
                --dtype float32 \
                --data_path_and_name_and_type "${_inference_dir}/wav.scp,speech,sound" \
                --valid_batch_size ${inference_batch_size} \
                --lid_train_config "${lid_exp}/config.yaml" \
                --lid_model_file "${lid_exp}"/${inference_model} \
                --use_preprocessor false \
                --fix_duration false \
                --num_workers ${nj} \
                --extract_embd ${extract_embd} \
                --checkpoint_interval ${checkpoint_interval} \
                --resume true \
                --save_embd_per_utt true \
                --save_embd_avg_lang false \
                --save_tsne_plot false

        touch ${infer_exp}/lid_and_embd_extract.done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Score on the test set."

    inference_model_name="${inference_model%.pth}"
    _lid_train_dir="${data_feats}/${train_set}"
    for test_set in ${test_sets_all}; do
        infer_exp="${lid_exp}/inference/${inference_model_name}/${test_set}"

        if [ -f "${infer_exp}/results" ] && grep -q "Accuracy" "${infer_exp}/results" && grep -q "Macro Accuracy" "${infer_exp}/results"; then
            log "Skip scoring for ${test_set} since it has already been done and is valid."
            continue
        fi
        pred_lids="${infer_exp}/${test_set}_lids"
        target_lids="${data_feats}/${test_set}/utt2lang"
        results="${infer_exp}/results"

        python ./local/score.py \
            --pred_lids "${pred_lids}" \
            --target_lids "${target_lids}" \
            --train_lang2utt "${_lid_train_dir}/lang2utt" \
            --results "${results}"
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Plot t-SNE."

    if [ -z "${tsne_set}" ]; then
        tsne_set="${train_set}"
    fi

    inference_model_name="${inference_model%.pth}"
    infer_exp="${lid_exp}/inference/${inference_model_name}/${tsne_set}"
    _inference_dir=${data_feats}/${tsne_set}

    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${infer_exp})"
    else
        jobname="${infer_exp}/inference_tsne.log"
    fi

    log "Plotting t-SNE... log: '${infer_exp}/inference_tsne.log'"
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log ${infer_exp}/inference_tsne.log \
        --ngpu ${ngpu} \
        --num_nodes ${num_nodes} \
        --init_file_prefix ${lid_exp}/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.lid_inference \
            --output_dir ${infer_exp} \
            --dtype float32 \
            --data_path_and_name_and_type "${_inference_dir}/wav.scp,speech,sound" \
            --data_path_and_name_and_type "${_inference_dir}/utt2lang,lid_labels,text" \
            --valid_batch_size ${inference_batch_size} \
            --lid_train_config "${lid_exp}/config.yaml" \
            --lid_model_file "${lid_exp}"/${inference_model} \
            --use_preprocessor true \
            --fix_duration false \
            --num_workers ${nj} \
            --extract_embd true \
            --checkpoint_interval ${checkpoint_interval} \
            --resume true \
            --save_embd_per_utt false \
            --save_embd_avg_lang true \
            --save_tsne_plot true \
            --max_utt_per_lang_for_tsne ${max_utt_per_lang_for_tsne} \
            --perplexity ${perplexity} \
            --max_iter ${max_iter}
fi

packed_model="${lid_exp}/${lid_exp##*/}_${inference_model%.*}.zip"
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ] && ! [[ " ${skip_stages} " =~ [[:space:]]9[[:space:]] ]]; then
    log "Stage 9: Pack model: ${packed_model}"

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.pack lid \
        --train_config "${lid_exp}"/config.yaml \
        --model_file "${lid_exp}"/"${inference_model}" \
        --option "${lid_exp}"/images \
        --outpath "${packed_model}"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ] && ! [[ " ${skip_stages} " =~ [[:space:]]10[[:space:]] ]]; then
    log "Stage 10: Upload model to HuggingFace: ${hf_repo}"
    [ -z "${hf_repo}" ] && \
        log "ERROR: You need to setup the variable hf_repo with the name of the repository located at HuggingFace, follow the following steps described here https://github.com/espnet/espnet/blob/master/CONTRIBUTING.md#132-espnet2-recipes" && \
    exit 1

    if [ ! -f "${packed_model}" ]; then
        log "ERROR: ${packed_model} does not exist. Please run stage 9 first."
        exit 1
    fi

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
    # /some/where/espnet/egs2/foo/lid1/ -> foo/lid1
    _task="$(pwd | rev | cut -d/ -f2 | rev)"
    # foo/asr1 -> foo
    _corpus="${_task%/*}"
    _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

    # copy files in ${dir_repo}
    unzip -o ${packed_model} -d ${dir_repo}
    # Generate description file
    # shellcheck disable=SC2034
    hf_task=language-identification
    # shellcheck disable=SC2034
    espnet_task=LID
    # shellcheck disable=SC2034
    task_exp=${lid_exp}
    # shellcheck disable=SC2034
    lang=multilingual
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
