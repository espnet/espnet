#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="val"
test_sets="eval"
# cls_config=conf/beats_cls.yaml
cls_config=conf/beats_cls_lightning.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
# mynametag=finl.EarLarge
mynametag=finl.beats

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"


## plz change the the wandb_entity
use_wandb=false
wandb_project=BEATsAS20K
wandb_args="--use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${mynametag} --wandb_entity shikhar"


./cls.sh \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --classification_type multi-label \
    --gpu_inference false \
    --use_lightning true \
    --feats_normalize uttmvn \
    --ngpu 2 \
    --stage 6 \
    --stop_stage 6 \
    --nj 10 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --inference_model valid.epoch_mAP.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --cls_args "${wandb_args}"
