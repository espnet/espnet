#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="development"
valid_set="validation"
test_sets="evaluation"
cls_config=conf/beats_entailment.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=${timestamp}
mynametag=tx_multimodal

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --speech_text_classification true \
    --text_input_filename hypothesis.txt \
    --hugging_face_model_name_or_path bert-base-uncased \
    --gpu_inference false \
    --feats_normalize uttmvn \
    --ngpu 2 \
    --stage 6 \
    --stop_stage 6 \
    --nj 10 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --inference_model valid.acc.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
