#!/usr/bin/env bash
# Train a Transformer LM warm-started from a public Chinese LM checkpoint.
# asr.sh's stage 7 has no --pretrained_lm passthrough, so we invoke
# espnet2.bin.launch + espnet2.bin.lm_train directly, mirroring how stage 7
# would have launched it. The resulting LM exp dir is then picked up by
# asr.sh during decode (stage 12) via --lm_exp / --lm_tag.
set -e
set -u
set -o pipefail

cd "$(dirname "$0")"
. ./path.sh

PRETRAINED_LM=${PRETRAINED_LM:-pretrained/magicdata_lm.pth}
LM_CONFIG=conf/train_lm_transformer_warmstart.yaml
EXP_DIR=exp/lm_train_lm_transformer_zh_char_warmstart
STATS_DIR=exp/lm_stats_zh_char        # reuse stats from the scratch LM run
NGPU=2

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

echo "=== LM warm-start training done. Now averaging best n checkpoints ==="
python3 -m espnet2.bin.aggregate_stats_dirs --help >/dev/null 2>&1 || true
# average the n-best models (mimic the trailing step of stage 7)
python3 -m espnet2.bin.average_nbest_models \
    --inputs "${EXP_DIR}"/{1..8}epoch.pth \
    --output "${EXP_DIR}/valid.loss.ave_5best.pth" \
    --num 5 \
    --best valid.loss \
    2>/dev/null || echo "(skipping manual average; trainer's auto-average should have produced valid.loss.ave_5best.pth)"
