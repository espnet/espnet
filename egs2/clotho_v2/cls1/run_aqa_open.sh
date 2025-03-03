#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="development_aqa_open"
valid_set="validation_aqa_open"
test_sets="evaluation_aqa_open"
cls_config=conf/beats_bert_multimodal_cls.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=${timestamp}
decoding_batch_size=20

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --local_data_opts "aqa open" \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data/aqa_open" \
    --dumpdir "${storage_dir}/dump/aqa_open" \
    --expdir "${storage_dir}/exp/aqa_open" \
    --speech_text_classification true \
    --text_input_filename question.txt \
    --hugging_face_model_name_or_path bert-base-uncased \
    --gpu_inference true \
    --decoding_batch_size ${decoding_batch_size} \
    --feats_normalize uttmvn \
    --ngpu 1 \
    --stage 1 \
    --stop_stage 10 \
    --nj 10 \
    --label_fold_length 600 \
    --inference_nj 1 \
    --inference_model valid.acc.best.pth \
    --cls_config "${cls_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
