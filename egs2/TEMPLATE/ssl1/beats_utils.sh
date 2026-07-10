# BEATs stage-7 helper functions, sourced by beats.sh.
# Split out to keep beats.sh readable. These run in beats.sh's shell, so they
# rely on its globals (expdir, ssl_tag, python, ...) and its log() helper.
# shellcheck shell=bash
# shellcheck disable=SC2154  # globals are provided by the sourcing beats.sh

setup_common_vars() {
    _ssl_train_dir="${data_feats}/${train_set}"
    _ssl_valid_dir="${data_feats}/${valid_set}"

    # Set tokenizer tag
    _tokenizer_inference_tag="tok"
    if [ -n "${tokenizer_inference_config}" ]; then
        _tokenizer_inference_tag="$(basename "${tokenizer_inference_config}" .yaml)"
    fi

    # Determine file types
    _feats_type="$(<${_ssl_train_dir}/feats_type)"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        _type=sound
        _waveform_opt="--waveform_input true"
    elif [ "${_feats_type}" = fbank ]; then
        _scp=feats.scp
        _type=kaldi_ark
        _waveform_opt="--waveform_input false"
    else
        log "Error: not supported: --feats_type ${feats_type}"
        exit 2
    fi
}


# Pick the best-model artifact saved by the ESPnet trainer. Tries the
# epoch-resolved symlinks first (best.pth, ave_1best.pth) and falls back
# to the multi-checkpoint average (ave_<N>best.pth) that the trainer emits
# when keep_nbest_models > 1 — loss first since SSL acc is noisy and
# tokenizer training only reports loss.
_best_ckpt_from_run_dir() {
    local run_dir_=$1
    local cand_
    for cand_ in valid.loss.best.pth valid.acc.best.pth \
                 valid.loss.ave_1best.pth valid.acc.ave_1best.pth \
                 valid.loss.ave.pth valid.acc.ave.pth; do
        if [[ -L "${run_dir_}/${cand_}" || -f "${run_dir_}/${cand_}" ]]; then
            echo "${run_dir_}/${cand_}"
            return 0
        fi
    done
    return 1
}

# Convert a trainer-saved checkpoint into the portable BEATs format used as
# teacher / tokenizer between iterations. Picks the best-valid artifact
# (per best_model_criterion). Auto-detects DeepSpeed vs regular trainer
# layout based on which artifact exists; falls back to the n-best average
# when the trainer didn't emit a per-epoch best symlink.
generate_checkpoint() {
    run_dir_=$1
    output_path_=$2

    best_ckpt_=$(_best_ckpt_from_run_dir "${run_dir_}")
    if [[ -z "${best_ckpt_}" ]]; then
        log "Error: Could not find best checkpoint in ${run_dir_}"
        return 1
    fi

    # If the artifact points to <N>epoch.pth, look up the matching DS or
    # plain checkpoint for that epoch. Otherwise (e.g. ave_<N>best.pth)
    # treat the file itself as the ESPnet state_dict.
    target_=$(readlink "${best_ckpt_}" 2>/dev/null) || target_=$(basename "${best_ckpt_}")
    target_=$(basename "${target_}")
    if [[ "${target_}" =~ ^([0-9]+)epoch\.pth$ ]]; then
        best_epoch_="${BASH_REMATCH[1]}"
        # DeepSpeed's save_checkpoint(dir, tag) writes under dir/tag/, so the
        # actual layout is checkpoint_<e>/<e>/mp_rank_00_model_states.pt.
        ds_ckpt_="${run_dir_}/checkpoint_${best_epoch_}/${best_epoch_}/mp_rank_00_model_states.pt"
        plain_ckpt_="${run_dir_}/${best_epoch_}epoch.pth"
        if [[ -f "${ds_ckpt_}" ]]; then
            log "Converting DeepSpeed checkpoint at epoch ${best_epoch_}: ${ds_ckpt_}"
            ${python} ../../../../espnet/espnet2/beats/generate_beats_checkpoint.py \
                --espnet_model_checkpoint_paths "${ds_ckpt_}" \
                --output_path "${output_path_}" \
                --espnet_model_config_path "${run_dir_}/config.yaml" \
                --deepspeed_checkpoint
        elif [[ -f "${plain_ckpt_}" ]]; then
            log "Converting ESPnet checkpoint at epoch ${best_epoch_}: ${plain_ckpt_}"
            ${python} ../../../../espnet/espnet2/beats/generate_beats_checkpoint.py \
                --espnet_model_checkpoint_paths "${plain_ckpt_}" \
                --output_path "${output_path_}" \
                --espnet_model_config_path "${run_dir_}/config.yaml"
        else
            log "Error: epoch ${best_epoch_} but no checkpoint at ${ds_ckpt_} or ${plain_ckpt_}"
            return 1
        fi
    else
        log "Converting averaged ESPnet checkpoint: ${best_ckpt_}"
        ${python} ../../../../espnet/espnet2/beats/generate_beats_checkpoint.py \
            --espnet_model_checkpoint_paths "${best_ckpt_}" \
            --output_path "${output_path_}" \
            --espnet_model_config_path "${run_dir_}/config.yaml"
    fi

    log "Checkpoint converted (source: ${best_ckpt_}) and stored at ${output_path_}"
}

train_encoder() {
    iteration=$1
    ssl_exp="${expdir}/beats_iter${iteration}_${ssl_tag}"

    log "Training encoder for iteration ${iteration}..."

    _opts=""
    [ -n "${train_config}" ] && _opts+="--config ${train_config} "

    if [ "${num_splits_ssl}" -gt 1 ]; then
        _split_dir="${ssl_stats_dir}/splits${num_splits_ssl}"
        _common_done="${_split_dir}/.done_common"
        _iter_done="${_split_dir}/.done_iter${iteration}"

        # Iteration-invariant files: split once and reuse across iterations.
        if [ ! -f "${_common_done}" ]; then
            ${python} -m espnet2.bin.split_scps \
                --scps "${_ssl_train_dir}/${_scp}" \
                "${ssl_stats_dir}/train/speech_shape" \
                "${ssl_stats_dir}/train/target_shape.word" \
                --num_splits "${num_splits_ssl}" \
                --output_dir "${_split_dir}"
            touch "${_common_done}"
        fi

        # Tokenized targets change each iteration; split per-iteration.
        # Same line order as the common files guarantees matching split assignment.
        if [ ! -f "${_iter_done}" ]; then
            ${python} -m espnet2.bin.split_scps \
                --scps "${_ssl_train_dir}/target_iter${iteration}_${_tokenizer_inference_tag}" \
                --num_splits "${num_splits_ssl}" \
                --output_dir "${_split_dir}"
            touch "${_iter_done}"
        fi

        _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text "
        _opts+="--train_shape_file ${_split_dir}/speech_shape "
        _opts+="--train_shape_file ${_split_dir}/target_shape.word "
        _opts+="--multiple_iterator true "
    else
        _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/${_scp},speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_ssl_train_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text "
        _opts+="--train_shape_file ${ssl_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${ssl_stats_dir}/train/target_shape.word "
    fi

    mkdir -p "${ssl_exp}"
    echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${ssl_exp}/run.sh"
    chmod +x "${ssl_exp}/run.sh"

    jobname=$(echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl && basename ${ssl_exp} || echo "${ssl_exp}/train.log")

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${ssl_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${ssl_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.beats_train \
            --use_preprocessor true \
            --token_list "${token_listdir}/tokens.txt" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/target_iter${iteration}_${_tokenizer_inference_tag},target,text" \
            --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${ssl_stats_dir}/valid/target_shape.word" \
            --resume true \
            --fold_length "${speech_fold_length}" \
            --fold_length "${text_fold_length}" \
            --output_dir "${ssl_exp}" \
            ${_opts} ${beats_args}

    # Generate float32 checkpoint after training completes
    checkpoint_path="${ssl_exp}/beats_encoder_iter${iteration}.pt"
    log "Generating float32 checkpoint from encoder training: ${checkpoint_path}"
    generate_checkpoint "${ssl_exp}" "${checkpoint_path}"
}

train_tokenizer() {
    iteration=$1
    ssl_tokenizer_exp="${expdir}/beats_tokenizer_iter${iteration}_${ssl_tag}"

    _opts=""
    [ -n "${tokenizer_train_config}" ] && _opts+="--config ${tokenizer_train_config} "

    # Setup teacher
    prev_iter=$((iteration - 1))
    prev_model_dir="${expdir}/beats_iter${prev_iter}_${ssl_tag}"
    teacher_ckpt_path_="${prev_model_dir}/beats_encoder_iter${prev_iter}.pt"

    if [ -n "${external_teacher_model}" ]; then
        teacher_ckpt_path_="${external_teacher_model}"
    else
        if [ ! -f "${teacher_ckpt_path_}" ]; then
            log "Generating teacher checkpoint from previous iteration"
            generate_checkpoint "${prev_model_dir}" "${teacher_ckpt_path_}"
        fi
    fi

    log "Training tokenizer for iteration ${iteration} using teacher: ${teacher_ckpt_path_}"

    mkdir -p "${ssl_tokenizer_exp}"
    echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${ssl_tokenizer_exp}/run.sh"
    chmod +x "${ssl_tokenizer_exp}/run.sh"

    jobname=$(echo "${cuda_cmd}" | grep -q -e queue.pl -e queue-freegpu.pl && basename ${ssl_tokenizer_exp} || echo "${ssl_tokenizer_exp}/train.log")

    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${ssl_tokenizer_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${ssl_tokenizer_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.beats_tokenizer_train \
            --use_preprocessor true \
            --beats_teacher_ckpt_path "${teacher_ckpt_path_}" \
            --train_data_path_and_name_and_type "${_ssl_train_dir}/${_scp},speech,${_type}" \
            --valid_data_path_and_name_and_type "${_ssl_valid_dir}/${_scp},speech,${_type}" \
            --train_shape_file "${ssl_stats_dir}/train/speech_shape" \
            --valid_shape_file "${ssl_stats_dir}/valid/speech_shape" \
            --resume true \
            --fold_length "${speech_fold_length}" \
            --output_dir "${ssl_tokenizer_exp}" \
            ${_opts} ${beats_args}

    # Convert tokenizer checkpoint to float32 for inference
    log "Generating float32 checkpoint from tokenizer training"
    checkpoint_path="${ssl_tokenizer_exp}/beats_tokenizer_iter${iteration}.pt"
    generate_checkpoint "${ssl_tokenizer_exp}" "${checkpoint_path}"
}

tokenizer_inference() {
    iteration=$1
    ssl_tokenizer_exp="${expdir}/beats_tokenizer_iter${iteration}_${ssl_tag}"

    _opts=""
    if [ -n "${external_tokenizer_model}" ]; then
        _opts+="--checkpoint_path ${external_tokenizer_model} "
    else
        tokenizer_checkpoint_path_="${ssl_tokenizer_exp}/beats_tokenizer_iter${iteration}.pt"
        if [ ! -f "${tokenizer_checkpoint_path_}" ]; then
            log "Generating tokenizer checkpoint for inference"
            generate_checkpoint "${ssl_tokenizer_exp}" "${tokenizer_checkpoint_path_}"
        fi
        _opts+="--checkpoint_path ${tokenizer_checkpoint_path_} "
        _opts+="--config_path ${ssl_tokenizer_exp}/config.yaml "
    fi
    _opts+="${_waveform_opt} "

    _nj=$((ngpu==0?nj:ngpu))
    _ngpu=$((ngpu==0?0:1))

    for _data_dir in "${_ssl_train_dir}" "${_ssl_valid_dir}"; do
        final_target_path_="${_data_dir}/target_iter${iteration}_${_tokenizer_inference_tag}"
        if [ -f "${final_target_path_}" ]; then
            log "Skipping tokenizer inference for ${_data_dir} as target already exists at ${final_target_path_}"
            continue
        fi
        ./scripts/feats/audio_tokenization.sh \
            --codec_choice beats \
            --file_name ${_scp} \
            --src_dir "${_data_dir}" \
            --tgt_dir "${_data_dir}/iter${iteration}_${_tokenizer_inference_tag}" \
            --nj "${_nj}" --ngpu "${_ngpu}" --batch_size "${tokenizer_inference_batch_size}" \
            ${_opts}

        cp "${_data_dir}/iter${iteration}_${_tokenizer_inference_tag}/${_scp%.scp}_beats.txt" \
            "${_data_dir}/target_iter${iteration}_${_tokenizer_inference_tag}"
    done
}
