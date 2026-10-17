#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

# Iteration range for BEATs iterative pretraining (see TEMPLATE/ssl1/beats.sh stage 7).
#   iter=0 : encoder trained against random-projection targets
#   iter=N (N>0) : tokenizer trained from encoder iter=N-1, then encoder iter=N trained against it
# train_start_iter=0, train_stop_iter=1 produces:
#   encoder iter1 -> tokenizer iter1 -> encoder iter2
train_start_iter=0
train_stop_iter=1

train_set="train"
valid_set="eval"

ssl_tag=base.audioset_2m

train_config=conf/beats_base.yaml
tokenizer_train_config=conf/tok_beats_base.yaml
tokenizer_inf_config=conf/as2m_inf.yaml

ngpu=2  # H200 x2 on Delta
storage_dir=.
mkdir -p "${storage_dir}"

# Stage map (see TEMPLATE/ssl1/beats.sh):
#   1-4 cpu : data prep, format, filter, fbank stats
#   5   gpu : tokenizer inference (per encoder iter > 0)
#   6   cpu : collect stats
#   7   gpu : encoder + tokenizer training loop
external_teacher_model=
external_tokenizer_model=

./beats.sh \
    --speech_fold_length 160000 \
    --text_fold_length 600 \
    --ssl_tag "${ssl_tag}" \
    --n_targets 1024 \
    --datadir "${storage_dir}/data" \
    --dumpdir "${storage_dir}/dump" \
    --expdir "${storage_dir}/exp" \
    --stage 1 \
    --stop_stage 7 \
    --feats_type fbank \
    --ngpu "${ngpu}" \
    --num_nodes 1 \
    --train_start_iter "${train_start_iter}" \
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 11 \
    --external_teacher_model "${external_teacher_model}" \
    --external_tokenizer_model "${external_tokenizer_model}" \
    --tokenizer_train_config "${tokenizer_train_config}" \
    --tokenizer_inference_config "${tokenizer_inf_config}" \
    --tokenizer_inference_batch_size 144 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --num_splits_ssl 1 \
    "$@"
