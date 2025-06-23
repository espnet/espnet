#!/usr/bin/env bash
set -euo pipefail

# Runs fine-tuning
asr_speech_fold_length=4800 # 480000/16000 = 30 seconds
pt_ckpt=valid.acc.ave_5best
inference_ckpt=latest

./asr.sh \
    --asr_tag ft \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 15 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 8 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set development_clotho_all \
    --valid_set validation \
    --test_sets "evaluation" \
    --asr_config conf/beats_bart_ft.yaml \
    --asr_speech_fold_length ${asr_speech_fold_length} \
    --pretrained_model exp/asr_pt/${pt_ckpt}.pth \
    --inference_asr_model ${inference_ckpt}.pth \
    --local_score_opts exp/asr_ft/inference_ctc_weight0.0_hugging_face_decoderTrue_asr_model_${inference_ckpt} \
    "$@"
