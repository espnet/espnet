#!/usr/bin/env bash
# Runs pre-training
set -euo pipefail

timestamp=$(date "+%Y%m%d.%H%M%S")

# Run pre-training

./asr.sh \
    --asr_tag pt.${timestamp} \
    --asr_speech_fold_length 1600 \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 15 \
    --ngpu 2 \
    --gpu_inference true \
    --nj 10 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set pretrain \
    --valid_set validation \
    --test_sets "validation evaluation" \
    --asr_config conf/beats_bart_pt.yaml \
    --inference_asr_model valid.acc.best.pth \
    --asr_args "${@}" \
    --local_score_opts "exp/asr_pt.${timestamp}/inference_beam_size10_ctc_weight0.0_hugging_face_decoderTrue_asr_model_valid.acc.best"



# Run fine-tuning

pt_tag=${timestamp}

ckpt_name=valid.acc.ave

pre_trained_model_path=${expdir}/asr_pt.${pt_tag}/${ckpt_name}.pth
ft_tag=${pt_tag}.${ckpt_name}


./asr.sh \
    --asr_tag ft.${ft_tag} \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 15 \
    --asr_stats_dir ${expdir}/asr_stats_finetune \
    --ngpu 2 \
    --gpu_inference true \
    --nj 8 \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --token_type hugging_face \
    --use_lm false \
    --hugging_face_model_name_or_path "facebook/bart-base" \
    --inference_args "--beam_size 10 --ctc_weight 0.0 --hugging_face_decoder True" \
    --train_set development_clotho \
    --valid_set validation \
    --test_sets "evaluation" \
    --asr_config conf/beats_bart_ft.yaml \
    --pretrained_model ${pre_trained_model_path} \
    --inference_asr_model valid.acc.ave_5best.pth \
    --asr_args "${wandb_init_args} ${other_args}" \
    --local_score_opts ${expdir}/asr_ft.${ft_tag}/inference_beam_size10_ctc_weight0.0_hugging_face_decoderTrue_asr_model_${ckpt_name}
