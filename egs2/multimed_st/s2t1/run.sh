#!/usr/bin/env bash
set -euo pipefail

src_lang=eng
tgt_lang=deu
task=st

max_train_samples=0
max_valid_samples=0
max_test_samples=0

lang=de
ngpu=0
nj=2
inference_nj=2
gpu_inference=false

train_set=train
valid_set=valid
test_sets="test"

# Expose common s2t.sh controls so parse_options.sh accepts them.
stage=1
stop_stage=100000
skip_data_prep=false
inference_s2t_model=valid.acc.ave.pth
inference_config=
download_model=
local_score_opts=

# Default: data-pipeline sanity mode.
token_type=char
nbpe=30
use_lm=false
s2t_config=
s2t_args=

# Optional: OWSM v4 small fine-tuning mode.
finetune_owsm_v4_small=false
owsm_max_epoch=1
owsm_num_iters_per_epoch=1000
owsm_batch_size=1
owsm_valid_batch_size=1
owsm_accum_grad=8

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if "${finetune_owsm_v4_small}"; then
    token_type=bpe
    nbpe=50000
    use_lm=false
    s2t_config=conf/finetune_owsm_v4_small.yaml
    inference_config=conf/decode_owsm_st_de.yaml
    s2t_args="--init_param downloads/owsm_v4_small_370M/model.pth --ignore_init_mismatch true"

    python local/prepare_owsm_v4_assets.py \
        --token_dir "data/${lang}_token_list/bpe_unigram${nbpe}" \
        --train_config "${s2t_config}" \
        --max_epoch "${owsm_max_epoch}" \
        --num_iters_per_epoch "${owsm_num_iters_per_epoch}" \
        --batch_size "${owsm_batch_size}" \
        --valid_batch_size "${owsm_valid_batch_size}" \
        --accum_grad "${owsm_accum_grad}"
fi

local_data_opts="--src_lang ${src_lang} --tgt_lang ${tgt_lang} --task ${task} --max_train_samples ${max_train_samples} --max_valid_samples ${max_valid_samples} --max_test_samples ${max_test_samples}"

./s2t.sh \
    --lang "${lang}" \
    --ngpu "${ngpu}" \
    --nj "${nj}" \
    --inference_nj "${inference_nj}" \
    --gpu_inference "${gpu_inference}" \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --skip_data_prep "${skip_data_prep}" \
    --inference_s2t_model "${inference_s2t_model}" \
    --inference_config "${inference_config}" \
    --download_model "${download_model}" \
    --token_type "${token_type}" \
    --nbpe "${nbpe}" \
    --use_lm "${use_lm}" \
    --s2t_config "${s2t_config}" \
    --s2t_args "${s2t_args}" \
    --local_score_opts "${local_score_opts}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --local_data_opts "${local_data_opts}"
