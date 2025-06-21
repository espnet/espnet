#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

# This is iteration 1 for icme bootstrapped with beats
train_start_iter=3
train_stop_iter=3
valid_set="eval"

train_set="train_soundmix"
ssl_stats_dir=/work/nvme/bbjs/sbharadwaj/icme_challenge/exp/beats_stats_fbank_soundmix

model_size=large

# train_set="train"
# ssl_stats_dir=/work/nvme/bbjs/sbharadwaj/icme_challenge/exp/beats_stats_fbank

tokenizer_inf_config=conf/beats_icme_inf.yaml

# ssl_tag=${model_size}.batch16.auris.icme
ssl_tag=${model_size}.batch16.steps2M.soundmix.auris.icme

if [ $model_size == "large" ]; then
    # train_config=conf/pretrain_large_icme.yaml
    train_config=conf/pretrain_auris_large_icme_cls_2msteps.yaml
    tokenizer_train_config=conf/tok_ear_large.yaml
elif [ $model_size == "base" ]; then
    train_config=conf/ear_base.yaml
    tokenizer_train_config=conf/tok_ear_base.yaml
else
    echo "Invalid model size"
    exit 1
fi

ngpu=4
nnodes=2

storage_dir=/work/nvme/bbjs/sbharadwaj/icme_challenge

mkdir -p "${storage_dir}"

# 1-4 : cpu (data prep: local, format, filter, fbank)
# 5: gpu (tokenization)
# 6: cpu (collect stats)
# 7: gpu (training)

num_splits_ssl=5
external_teacher_model=
external_tokenizer_model=


use_wandb=true
wandb_project=ICME
wandb_args="--use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${ssl_tag} --wandb_entity shikhar"

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag ${ssl_tag} \
    --ssl_stats_dir "${ssl_stats_dir}" \
    --n_targets 1024 \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --stage 7 \
    --stop_stage 7 \
    --feats_type fbank \
    --ngpu ${ngpu} \
    --num_nodes ${nnodes} \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 128 \
    --max_wav_duration 11 \
    --external_teacher_model "${external_teacher_model}" \
    --external_tokenizer_model "${external_tokenizer_model}" \
    --tokenizer_train_config "${tokenizer_train_config}" \
    --tokenizer_inference_config "${tokenizer_inf_config}" \
    --tokenizer_inference_batch_size 144 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --num_splits_ssl "${num_splits_ssl}" \
    --beats_args "${wandb_args}"


# ./beats.sh \
#     --train_set train \
#     --valid_set eval \
#     --tokenizer_inference_config conf/beats_icme_inf.yaml \
#     --num_splits_ssl 5 \
#     --stage 6 \
#     --stop_stage 6 \
#     --n_targets 1024 \
#     --tokenizer_inference_batch_size 700 \
#     --speech_fold_length 160000 \
#     --text_fold_length 600 \
#     --ssl_tag icme_data \
#     --datadir /work/nvme/bbjs/sbharadwaj/fullas2m/data \
#     --dumpdir /work/nvme/bbjs/sbharadwaj/icme_challenge/dump \
#     --expdir /work/nvme/bbjs/sbharadwaj/icme_challenge/exp \
#     --audio_format wav \
#     --fs 16k \
#     --nj 128 \
#     --ngpu 64 \
#     --min_wav_duration 0.1 \
#     --max_wav_duration 11 \
#     --feats_type fbank \
#     --train_start_iter 0 \
#     --train_stop_iter 0 \
#     --external_tokenizer_model /work/nvme/bbjs/sbharadwaj/model_checkpoints/BEATs_ckpt/Tokenizer_iter3.pt \
#     --train_config conf/pretrain_large_icme.yaml