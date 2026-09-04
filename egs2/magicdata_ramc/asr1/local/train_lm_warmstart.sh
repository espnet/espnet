#!/usr/bin/env bash
# Recipe-local helper: train a Transformer LM warm-started from a public
# Chinese LM checkpoint.
#
# asr.sh's stage 7 has no --pretrained_lm passthrough, so we invoke
# espnet2.bin.launch + espnet2.bin.lm_train directly, mirroring how stage 7
# would have launched it. The resulting LM exp dir is then picked up by
# asr.sh during decode (stage 12) via --lm_exp.
#
# Run from the recipe root. The default PRETRAINED_LM points at the magicdata
# LM ckpt; download it to that path first, e.g.:
#   wget -O pretrained/magicdata_lm.pth \
#     https://huggingface.co/espnet/jiyangtang_magicdata_asr_conformer_lm_transformer/resolve/main/exp/lm_train_lm_transformer_zh_char/valid.loss.ave_10best.pth
#
# Then to decode with this LM, run e.g.:
#   ./run.sh --lm_exp exp/lm_train_lm_transformer_zh_char_warmstart \
#            --stage 12 --stop-stage 13
set -e
set -u
set -o pipefail

. ./path.sh

PRETRAINED_LM=${PRETRAINED_LM:-pretrained/magicdata_lm.pth}
LM_CONFIG=conf/train_lm_transformer_warmstart.yaml
EXP_DIR=exp/lm_train_lm_transformer_zh_char_warmstart
STATS_DIR=exp/lm_stats_zh_char    # reuse stats from a prior scratch-LM stage-6 run
NGPU=2

if [ ! -f "${PRETRAINED_LM}" ]; then
    echo "Error: pretrained LM checkpoint not found at ${PRETRAINED_LM}" >&2
    echo "Download it first or set PRETRAINED_LM=<path> in the environment." >&2
    exit 1
fi
if [ ! -d "${STATS_DIR}" ]; then
    echo "Error: ${STATS_DIR} not found." >&2
    echo "Run './run.sh --stage 6 --stop-stage 6' first to produce LM stats." >&2
    exit 1
fi

mkdir -p "${EXP_DIR}"

python3 -m espnet2.bin.launch \
    --cmd "run.pl --name ${EXP_DIR}/train.log" \
    --log "${EXP_DIR}/train.log" \
    --ngpu ${NGPU} \
    --num_nodes 1 \
    --init_file_prefix "${EXP_DIR}/.dist_init_" \
    --multiprocessing_distributed true \
    -- \
    python3 -m espnet2.bin.lm_train \
        --ngpu ${NGPU} \
        --use_preprocessor true \
        --bpemodel none \
        --token_type char \
        --token_list data/zh_token_list/char/tokens.txt \
        --non_linguistic_symbols none \
        --cleaner none \
        --g2p none \
        --valid_data_path_and_name_and_type dump/raw/org/dev/text,text,text \
        --valid_shape_file "${STATS_DIR}/valid/text_shape.char" \
        --fold_length 150 \
        --resume true \
        --init_param "${PRETRAINED_LM}" \
        --ignore_init_mismatch true \
        --output_dir "${EXP_DIR}" \
        --config "${LM_CONFIG}" \
        --train_data_path_and_name_and_type dump/raw/lm_train.txt,text,text \
        --train_shape_file "${STATS_DIR}/train/text_shape.char"

echo "Done. The trainer's auto-averaging will have produced ${EXP_DIR}/valid.loss.ave.pth"
