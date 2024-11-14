#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_SKA_mel_1_100.yaml

train_set="asvspoof5_train"
valid_set="dev"
cohort_set="dev"
test_sets="eval"
skip_train=false

feats_type="raw"

ngpu=1
nj=8
speed_perturb_factors=
audio_format=flac
inference_model=valid.a_dcf.best.pth
pretrained_model=9epoch.pth
ignore_init_mismatch=true

cos_sim=true
multi_task=true
sasv_task=true
embed_avg=true
no_labels=true 

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --cohort_set ${cohort_set} \
    --test_sets ${test_sets} \
    --skip_train ${skip_train} \
    --ngpu ${ngpu} \
    --nj ${nj} \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --audio_format ${audio_format} \
    --inference_model ${inference_model} \
    --pretrained_model ${pretrained_model} \
    --ignore_init_mismatch ${ignore_init_mismatch} \
    --cos_sim ${cos_sim} \
    --multi_task ${multi_task} \
    --sasv_task ${sasv_task} \
    --embed_avg ${embed_avg} \
    --no_labels ${no_labels} \
    "$@"
