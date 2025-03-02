#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="enabirds.train"
valid_set="enabirds.dev"
test_sets="enabirds.test"
cls_config=conf/beats_dcase.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
#timestamp=20250127.164810
mynametag=dcase.${timestamp}
storage_dir=/raid/users/popcornell/ESPnet_BEATS/espnet/egs2/beans/cls1/beats_run_enabirds
mkdir -p "${storage_dir}"

#change label_fold_length for each dataset: greater than num_class

./cls.sh \
    --local_data_opts "enabirds" \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --feats_normalize uttmvn \
    --stage 1 \
    --stop_stage 10 \
    --ngpu 1 \
    --gpu_inference true \
    --nj 10 \
    --speech_fold_length 160000 \
    --label_fold_length 35 \
    --max_wav_duration 32 \
    --inference_nj 1 \
    --inference_model valid.mAP.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"