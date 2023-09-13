#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PATH=$PATH:/tmp2/anthony/ffmpeg-6.0-amd64-static
export CUDNN_PATH=/tmp2/anthony/anthony_espnet/espnet/tools/miniconda/envs/dgslm/lib/python3.8/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_PATH/lib
export PYTHONPATH=/tmp2/anthony/anthony_espnet/espnet
kmeans_feature="hubert_custom/21"  # use model_type/layer_index
nclusters=2000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=ch

set=L    # S for the small set, M for the mediate set, L for the large set

train_set=train_"$(echo "${set}" | tr "[:upper:]" "[:lower:]")"
valid_set=dev
test_sets="dev test_meeting test_net"

asr_config=conf/train_discrete_asr_e_branchformer1.yaml
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=6000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --stage 1 \
    --stop_stage 12 \
    --skip_stages 8,9,10,11 \
    --kmeans_opts "--batch_bins 4800000 --portion 0.01 --storage_save_mode true" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --ngpu 4 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --local_data_opts "--set ${set}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --num_splits_asr 8 \
    --nj 32
    # --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang} data/local/other_text/text" "$@"
    # --speed_perturb_factors "0.9 1.0 1.1" \
