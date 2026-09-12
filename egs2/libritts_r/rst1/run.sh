#!/usr/bin/env bash
set -euo pipefail

. ./path.sh
. ./cmd.sh
. ./db.sh

stage=1
stop_stage=11
ngpu=4
nj=64
python=python3
config=conf/train.yaml
decode_config=conf/decode.yaml
expdir=exp/sidon_w2v_bert2_layer8
# Vocoder (stages 6-8). Stage 7 pretrains on ground-truth features, stage 8
# finetunes on the stage-5 predictor's features. The config's vocoder_type
# picks the DAC decoder (default) or ESPnet's HiFi-GAN generator.
voc_pretrain_config=conf/tuning/train_sidon_vocoder_pretrain.yaml
voc_finetune_config=conf/tuning/train_sidon_vocoder_finetune.yaml
voc_pretrain_exp=exp/sidon_vocoder_pretrain
voc_finetune_exp=exp/sidon_vocoder_finetune
# Warm start for stage 8: default the stage-7 best. To start from the
# published vocoder instead, run local/convert_official_sidon_vocoder.py and
# pass --vocoder_init exp/official_sidon_vocoder/vocoder.pth
# --discriminator_init "" (the release has no discriminator).
vocoder_init=
discriminator_init=
# Vocoder used at inference: an ESPnet-trained one (default the stage-8
# best) or, if --sidon_vocoder is set, the official TorchScript decoder.
vocoder_exp=
vocoder_model_file=
sidon_vocoder=
test_sets="test-clean test-other"
versa_config=conf/versa_enh.yaml
versa_ref_config=conf/versa_enh_ref_based.yaml
# Optional clean reference for synthetically degraded inputs.  The placeholder
# {test_set} is replaced per evaluation set.
ref_wav_scp=

. utils/parse_options.sh

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*"; }


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: data preparation"
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: resample feature-predictor data to 16 kHz"
    for split in train dev; do
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" --cmd "${train_cmd}" --fs 16000 --audio-format wav \
            "data/${split}_fp/wav.scp" "data/${split}_fp_16k"
    done
    for test_set in ${test_sets}; do
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" --cmd "${decode_cmd}" --fs 16000 --audio-format wav \
            "data/${test_set}/wav.scp" "data/${test_set}_16k"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: generate RIR pool"
    ${python} local/prepare_rir_pool.py \
        --out_dir data/rir_pool --n_rirs 50000 --nj ${nj}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: collect feature-predictor statistics"
    ${python} -m espnet2.bin.rst_train \
        --config ${config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --output_dir ${expdir} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: train feature predictor"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        ${python} -m espnet2.bin.rst_train \
        --config ${config} \
        --train_data_path_and_name_and_type data/train_fp_16k/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_fp_16k/wav.scp,speech_ref1,sound \
        --train_shape_file ${expdir}/train/speech_ref1_shape \
        --valid_shape_file ${expdir}/valid/speech_ref1_shape \
        --output_dir ${expdir} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: collect vocoder statistics"
    ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --output_dir ${voc_pretrain_exp} --collect_stats true --ngpu 0
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: pretrain vocoder on ground-truth SSL features"
    ${cuda_cmd} --gpu ${ngpu} ${voc_pretrain_exp}/train.log \
        ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_pretrain_config} \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --train_shape_file ${voc_pretrain_exp}/train/speech_ref1_shape \
        --valid_shape_file ${voc_pretrain_exp}/valid/speech_ref1_shape \
        --output_dir ${voc_pretrain_exp} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: finetune vocoder on predicted SSL features"
    vocoder_init=${vocoder_init:-${voc_pretrain_exp}/valid.loss_mel.best.pth}
    discriminator_init=${discriminator_init-${vocoder_init}}
    init_opts=(--init_param "${vocoder_init}:vocoder:vocoder")
    if [ -n "${discriminator_init}" ]; then
        init_opts+=(--init_param "${discriminator_init}:discriminator:discriminator")
    fi
    # Same utterances as stage 7, so its shape files are reused.
    ${cuda_cmd} --gpu ${ngpu} ${voc_finetune_exp}/train.log \
        ${python} -m espnet2.bin.rst_vocoder_train \
        --config ${voc_finetune_config} \
        --fp_model_path ${expdir}/valid.loss.best.pth \
        "${init_opts[@]}" \
        --train_data_path_and_name_and_type data/train_voc/wav.scp,speech_ref1,sound \
        --valid_data_path_and_name_and_type data/dev_voc/wav.scp,speech_ref1,sound \
        --train_shape_file ${voc_pretrain_exp}/train/speech_ref1_shape \
        --valid_shape_file ${voc_pretrain_exp}/valid/speech_ref1_shape \
        --output_dir ${voc_finetune_exp} --ngpu ${ngpu} \
        --multiprocessing_distributed true --unused_parameters true --resume true
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    if [ -n "${sidon_vocoder}" ]; then
        vocoder_opts=(--sidon_vocoder "${sidon_vocoder}")
    else
        vocoder_exp=${vocoder_exp:-${voc_finetune_exp}}
        vocoder_model_file=${vocoder_model_file:-${vocoder_exp}/valid.loss_mel.best.pth}
        for required_file in "${vocoder_exp}/config.yaml" "${vocoder_model_file}"; do
            [ -f "${required_file}" ] || {
                log "Missing vocoder file ${required_file}: train one (stages 6-8) or set --sidon_vocoder"
                exit 1
            }
        done
        vocoder_opts=(--vocoder_train_config "${vocoder_exp}/config.yaml"
                      --vocoder_model_file "${vocoder_model_file}")
    fi
    for test_set in ${test_sets}; do
        log "Stage 9: inference (${test_set})"
        ${python} -m espnet2.bin.rst_inference \
            --config ${decode_config} \
            --train_config ${expdir}/config.yaml \
            --model_file ${expdir}/valid.loss.best.pth \
            "${vocoder_opts[@]}" \
            --wav_scp data/${test_set}_16k/wav.scp \
            --output_dir ${expdir}/inference_${test_set}
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    for test_set in ${test_sets}; do
        # The paper's four metrics without VERSA; stage 11 (VERSA) covers them
        # and more, so this stage can be skipped when VERSA is installed.
        log "Stage 10: scoring (${test_set})"
        text_opt=()
        if [ -f "data/${test_set}/text" ]; then
            text_opt=(--text "data/${test_set}/text")
        fi
        ${python} local/score.py \
            --restored_dir ${expdir}/inference_${test_set}/wav \
            --ref_wav_scp data/${test_set}/wav.scp \
            --noisy_wav_scp data/${test_set}_16k/wav.scp \
            "${text_opt[@]}" \
            --output_dir ${expdir}/score_${test_set}
    done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    ${python} -c "import versa" || {
        log "VERSA is required for stage 11; run tools/installers/install_versa.sh"
        exit 1
    }
    for test_set in ${test_sets}; do
        log "Stage 11: VERSA scoring (${test_set})"
        inf_dir=${expdir}/inference_${test_set}
        eval_dir=${inf_dir}/scoring/versa_eval
        pred_scp=${inf_dir}/wav.scp
        input_scp=data/${test_set}_16k/wav.scp
        text=data/${test_set}/text

        for required_file in "${pred_scp}" "${input_scp}" "${text}"; do
            [ -f "${required_file}" ] || {
                log "Missing VERSA input: ${required_file}"
                exit 1
            }
        done
        mkdir -p "${eval_dir}"
        num_pred=$(wc -l < "${pred_scp}")
        score_nj=$(( nj < num_pred ? nj : num_pred ))
        [ "${score_nj}" -gt 0 ] || { log "No inference output to score"; exit 1; }

        split_pred=()
        for n in $(seq "${score_nj}"); do
            split_pred+=("${eval_dir}/pred.${n}")
        done
        utils/split_scp.pl "${pred_scp}" "${split_pred[@]}"

        ${decode_cmd} JOB=1:"${score_nj}" "${eval_dir}/versa.JOB.log" \
            ${python} -m versa.bin.scorer \
                --pred "${eval_dir}/pred.JOB" \
                --gt "${input_scp}" \
                --text "${text}" \
                --score_config "${versa_config}" \
                --cache_folder "${eval_dir}/cache" \
                --output_file "${eval_dir}/result.JOB.txt" \
                --io soundfile
        ${python} pyscripts/utils/aggregate_eval.py \
            --logdir "${eval_dir}" --scoredir "${eval_dir}" --nj "${score_nj}"

        if [ -n "${ref_wav_scp}" ]; then
            ref_scp=${ref_wav_scp//\{test_set\}/${test_set}}
            [ -f "${ref_scp}" ] || { log "Missing clean reference: ${ref_scp}"; exit 1; }
            ref_dir=${inf_dir}/scoring/versa_ref
            mkdir -p "${ref_dir}"
            ${decode_cmd} JOB=1:"${score_nj}" "${ref_dir}/versa.JOB.log" \
                ${python} -m versa.bin.scorer \
                    --pred "${eval_dir}/pred.JOB" \
                    --gt "${ref_scp}" \
                    --score_config "${versa_ref_config}" \
                    --cache_folder "${ref_dir}/cache" \
                    --output_file "${ref_dir}/result.JOB.txt" \
                    --io soundfile
            ${python} pyscripts/utils/aggregate_eval.py \
                --logdir "${ref_dir}" --scoredir "${ref_dir}" --nj "${score_nj}"
        else
            log "Skipping reference-based VERSA metrics"
        fi
    done
fi
