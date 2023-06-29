#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export PATH=~/rds/rds-t2-cs164-KQ4S3rlDzm8/gs534/MultiModal/espnet/tools/sctk/bin/:$PATH

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_rnnt.yaml
inference_config=conf/decode_asr.yaml
asr_tag=train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep

./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 600 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --asr_tag ${asr_tag} \
    --inference_asr_model valid.loss.ave.pth \
    --biasing true \
    --bpe_train_text "data/${train_set}/text" "$@"
