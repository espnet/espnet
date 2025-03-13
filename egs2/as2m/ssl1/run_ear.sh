#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=1
train_stop_iter=1

train_set="train"
valid_set="eval"

timestamp=$(date "+%m%d.%H%M%S")
mynametag=

ssl_tag=${mynametag}.${timestamp}


# tokenizer_inf_config=conf/tokenizer_inference_beats2.yaml
tokenizer_inf_config=conf/tokenizer_large_100k_steps.yaml
model_size=large
ssl_tag=${model_size}.7p2M

if [ $model_size == "large" ]; then
    train_config=conf/ear_large.yaml
    tokenizer_train_config=conf/tok_ear_large.yaml
elif [ $model_size == "base" ]; then
    train_config=conf/ear_base.yaml
    tokenizer_train_config=conf/tok_ear_base.yaml
else
    echo "Invalid model size"
    exit 1
fi
ngpu=4

storage_dir=.
storage_dir=/work/nvme/bbjs/sbharadwaj/7Msounds
mkdir -p "${storage_dir}"

# 1-4 : cpu (data prep: local, format, filter, fbank)
# 5: gpu (tokenization)
# 6: cpu (collect stats)
# 7: gpu (training)
num_splits_ssl=1
external_teacher_model=
external_tokenizer_model=
# use_wandb=true
# wandb_project=BEATsPTi0
# wandb_args="--use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${ssl_tag} --wandb_entity shikhar"

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag ${ssl_tag} \
    --n_targets 1024 \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --stage 7 \
    --stop_stage 7 \
    --feats_type fbank \
    --ngpu ${ngpu} \
    --num_nodes 1 \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 11 \
    --external_teacher_model "${external_teacher_model}" \
    --external_tokenizer_model "${external_tokenizer_model}" \
    --tokenizer_train_config "${tokenizer_train_config}" \
    --tokenizer_inference_config "${tokenizer_inf_config}" \
    --tokenizer_inference_batch_size 160 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --num_splits_ssl "${num_splits_ssl}" \
    "$@" #--beats_args "${wandb_args}"
