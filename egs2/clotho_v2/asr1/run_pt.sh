#!/usr/bin/env bash
set -euo pipefail

# Runs pre-training
asr_speech_fold_length=4800 # 480000/16000 = 30 seconds
inference_ckpt=valid.acc.ave_5best


./asr.sh \
    --asr_tag pt \
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
    --train_set pretrain \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --asr_config conf/beats_bart_pt.yaml \
    --asr_speech_fold_length ${asr_speech_fold_length} \
    --inference_asr_model ${inference_ckpt}.pth \
    --local_score_opts "exp/asr_pt/inference_ctc_weight0.0_hugging_face_decoderTrue_asr_model_${inference_ckpt}" \
    "$@"