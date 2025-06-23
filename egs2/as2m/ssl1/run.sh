#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=1
train_stop_iter=3

train_set="train"
valid_set="eval"

timestamp=$(date "+%m%d.%H%M%S")
ssl_tag=${timestamp}

tokenizer_inf_config=conf/7Msounds_inf.yaml
model_size=large
ssl_tag=${model_size}.7p2M

if [ $model_size == "large" ]; then
    train_config=conf/openbeats_large.yaml
    tokenizer_train_config=conf/tok_openbeats_large.yaml
elif [ $model_size == "base" ]; then
    train_config=conf/openbeats_base.yaml
    tokenizer_train_config=conf/tok_openbeats_base.yaml
else
    echo "Invalid model size"
    exit 1
fi
ngpu=1

storage_dir=.
mkdir -p "${storage_dir}"

# 1-4 : cpu (data prep: local, format, filter, fbank)
# 5: gpu (tokenization)
# 6: cpu (collect stats)
# 7: gpu (training)
num_splits_ssl=1
external_teacher_model=
external_tokenizer_model=

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag ${ssl_tag} \
    --n_targets 1024 \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --stage 1 \
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
    "$@"
