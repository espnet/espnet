#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

norm="" # or "_unnorm"
norm="_unnorm"
train_set=train_nodev${norm}
valid_set=dev_4k${norm}
test_sets="dev_4k${norm} val${norm}"

asr_config=conf/tuning/train_asr_conformer6_n_fft512_hop_length256.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=""

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --ngpu 4 \
    --nj 128 \
    --inference_nj 256 \
    --dumpdir dump${norm} \
    --expdir exp${norm} \
    --lang en${norm} \
    --token_type bpe \
    --nbpe 5000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --score_opts "-s" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
