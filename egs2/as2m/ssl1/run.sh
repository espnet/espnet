#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=0

train_set="train"
valid_set="eval"

timestamp=$(date "+%m%d.%H%M%S")
mynametag=

ssl_tag=${mynametag}.${timestamp}


# ssl_tag=ordered.deepnorm.relpos_noflash.default_hyp.0207.155136
# ssl_tag=ordered.deepnorm.relpos_noflash.prenorm_pred.default_hyp.0207.161949

# ssl_tag=xv_normalcodes.noflash.0212.174746
# modelsize=small

i=2
ssl_tag=bs2k.test_configurations.base.xv_normal$i
modelsize=base
tokenizer_inf_config=conf/tokenizer_inference_beats$i.yaml

if [ "${modelsize}" == "large" ]; then
    train_config=conf/pretrain_beats_large_as2m.yaml
    ngpu=3
else
    train_config=conf/tuning_bs_pretrain_beats_as2m.yaml
    ngpu=4
fi

storage_dir=.
# storage_dir=/work/nvme/bbjs/sbharadwaj/7Msounds
storage_dir=/work/nvme/bbjs/sbharadwaj/fullas2m
mkdir -p "${storage_dir}"

# 1-4 : cpu (data prep: local, format, filter, fbank)
# 5: gpu (tokenization)
# 6: cpu (collect stats)
# 7: gpu (training)

use_wandb=true
wandb_project=BEATsPTi0
wandb_entity=shikhar


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
    --tokenizer_inference_config "${tokenizer_inf_config}" \
    --tokenizer_inference_batch_size 160 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --beats_args "--use_wandb ${use_wandb} --wandb_project ${wandb_project} --wandb_name ${ssl_tag} --wandb_entity ${wandb_entity}" \
    "$@"
