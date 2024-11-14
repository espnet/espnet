#!/usr/bin/env bash
set -euo pipefail
# Runs fine-tuning

# wandb_init_args="--use_wandb true --wandb_project DCASE_AAC --wandb_model_log_interval 0"
wandb_init_args=""
other_args="$@"

# expdir=/compute/babel-13-33/sbharad2/expdir
# dumpdir=/compute/babel-13-33/sbharad2/dumpdir
# local_data_opts=/compute/babel-13-33/sbharad2/expdir
#     --expdir ${expdir} \
#     --dumpdir ${dumpdir} \
#     --local_data_opts "${local_data_opts}" \

# pre_trained_model_path=exp/asr_pt.initfix.bigbatch512.lr2e-4.weighted12layers.20241103.145125/valid.acc.ave_5best.pth
# pt_tag=full.20241111.170406
# pt_tag=data_check.20241111.153210

pt_tag=beats.bartfrozen.20241111.164042

ckpt_name=valid.acc.ave

pre_trained_model_path=${expdir}/asr_pt.${pt_tag}/${ckpt_name}.pth
ft_tag=${pt_tag}.${ckpt_name}


./asr.sh \
    --asr_tag ft.${ft_tag} \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 13 \
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
