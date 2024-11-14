#!/usr/bin/env bash
# Runs pre-training
set -euo pipefail

# # Run inference on all checkpoints

wandb_sweep_args="$@"


# expdir=/compute/babel-13-33/sbharad2/expdir
# dumpdir=/compute/babel-13-33/sbharad2/dumpdir
#         --expdir ${expdir} \
#         --dumpdir ${dumpdir} \


name_tag=ft.beats.bartfrozen.20241111.164042.valid.acc.ave
all_ckpts=$(find ${expdir}/asr_${name_tag} -name "*.pth"  -printf "%f\n" | awk '{print substr($0, 1, length($0) - 4)}')

# for ckpt in valid.acc.best; do
for ckpt in $all_ckpts; do
    # We skip all symlinks
    if [[ $ckpt == "checkpoint" || $ckpt == "latest"  || $ckpt == "valid.acc.best"  || $ckpt == "valid.acc.ave.pth" ]]; then
        continue
    fi
    ./asr.sh \
        --asr_tag $name_tag \
        --feats_normalize uttmvn \
        --stage 12 \
        --stop_stage 13 \
        --ngpu 2 \
        --gpu_inference true \
        --nj 20 \
        --inference_nj 1 \
        --max_wav_duration 30 \
        --token_type hugging_face \
        --use_lm false \
        --hugging_face_model_name_or_path "facebook/bart-base" \
        --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
        --train_set pretrain \
        --valid_set validation \
        --test_sets "evaluation" \
        --inference_asr_model ${ckpt}.pth \
        --asr_args "${wandb_sweep_args}" \
        --local_score_opts "${expdir}/asr_${name_tag}/inference_beam_size10_ctc_weight0.0_hugging_face_decoderTrue_asr_model_${ckpt}"
done