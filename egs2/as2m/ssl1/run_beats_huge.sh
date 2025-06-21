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

tokenizer_inf_config=conf/tokenizer_huge_inf.yaml
model_size=huge
ssl_tag=${model_size}.7p2M

train_config=conf/beats_huge.yaml
tokenizer_train_config=conf/beats_tok_huge.yaml
ngpu=8

storage_dir=.
storage_dir=/work/nvme/bbjs/sbharadwaj/7Msounds
# storage_dir=/work/nvme/bbjs/sbharadwaj/7100Ksounds
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
    --n_targets 4096 \
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
    "$@" #--beats_args "${wandb_args}"
