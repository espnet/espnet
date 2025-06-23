#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="development_cle"
valid_set="validation_cle"
test_sets="evaluation_cle"
cls_config=conf/beats_bert_multimodal_cls.yaml

timestamp=$(date "+%Y%m%d.%H%M%S")
mynametag=cle.${timestamp}
decoding_batch_size=20

storage_dir=. # change this to where you have space, if needed
mkdir -p "${storage_dir}"

./cls.sh \
    --local_data_opts "entailment" \
    --cls_tag "${mynametag}" \
    --datadir "${storage_dir}/data/cle" \
    --dumpdir "${storage_dir}/dump/cle" \
    --expdir "${storage_dir}/exp/cle" \
    --speech_text_classification true \
    --text_input_filename hypothesis.txt \
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
