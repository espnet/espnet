#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# The restoration recipe (rst.sh -> egs2/libritts_r/rst1/run.sh) on mini_an4
# with tiny models, for CI. Stages 10-11 (scoring) download large evaluation
# models and are left out by default.
./rst.sh \
    --ngpu 0 \
    --nj 2 \
    --n_rirs 4 \
    --fp_config conf/train_rst_debug.yaml \
    --decode_config conf/decode_debug.yaml \
    --expdir exp/rst_debug \
    --voc_pretrain_config conf/train_rst_vocoder_pretrain_debug.yaml \
    --voc_finetune_config conf/train_rst_vocoder_finetune_debug.yaml \
    --voc_pretrain_exp exp/rst_vocoder_pretrain_debug \
    --voc_finetune_exp exp/rst_vocoder_finetune_debug \
    --test_sets "test" \
    --stop_stage 9 "$@"
