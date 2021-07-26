#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=2
stop_stage=3
ngpu=4

# pretrain_mfcc_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain.yaml
pretrain_mfcc_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain_gpu32.yaml
pretrain_iter1_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain_it1.yaml
pretrain_iter2_config=conf/tuning/train_asr_hubert_base_960h_full_pretrain_it2.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml # didnt' use
inference_config=conf/decode_asr.yaml

mfcc_n_clusters=100
hubert_n_clusters=500
feature_mfcc="mfcc"
feature_hubert_iter1="HuBERT6"
feature_hubert_iter2="HuBERT9"

log "$0 $*"
. utils/parse_options.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${test_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # sbatch --export=PATH  --time 48:00:00 -p RM-shared --mem-per-cpu 32G -c 1 --open-mode=append \
    #     -e log/km_stage2.log -o log/km_stage2.log \
    #     run_pretrain.sh --stage 1 --stop-stage 1
    log "Stage 2: Running K-means on MFCC feature."

    ./local/km.sh \
        --stage 1 --stop-stage 3 \
        --nclusters ${mfcc_n_clusters} \
        --feature-type ${feature_mfcc} \
        --datadir "./data" \
        --kmrootdir "./exp" \
        --dictdir "./data/${feature_mfcc}_km${mfcc_n_clusters}_token_list/word"
fi

train_set="train_960_${feature_mfcc}_km${mfcc_n_clusters}"
valid_set="dev_${feature_mfcc}_km${mfcc_n_clusters}"
test_sets="test_clean_${feature_mfcc}_km${mfcc_n_clusters} \
    test_other_${feature_mfcc}_km${mfcc_n_clusters} \
    dev_clean_${feature_mfcc}_km${mfcc_n_clusters} \
    dev_other_${feature_mfcc}_km${mfcc_n_clusters}"

#TODO(use asr.sh stage 4 to dump data)
mkdir -p dump/${feature_mfcc}_km${mfcc_n_clusters}/raw
for task in ${train_set} ${valid_set} ${test_sets}; do
    cp -r data/${task} dump/${feature_mfcc}_km${mfcc_n_clusters}/raw
    echo "raw" > dump/${feature_mfcc}_km${mfcc_n_clusters}/raw/${task}/feats_type
done

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Pretrain HuBERT model using MFCC K-means pseudo-labels."

    ./hubert_asr.sh \
        --stage 10 --stop-stage 10 \
        --lang ${feature_mfcc}_km${mfcc_n_clusters} \
        --ngpu ${ngpu} \
        --nj 32 \
        --max_wav_duration 30 \
        --asr_config "${pretrain_mfcc_config}" \
        --dumpdir "dump/${feature_mfcc}_km${mfcc_n_clusters}" \
        --train-set "${train_set}" \
        --valid-set "${valid_set}" \
        --test_sets "${test_sets}" \
        --feats-normalize null \
        --token_type word "$@"
fi

# Secend iteration, extract feature from hubert layer6 and re-generate label
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Running K-means on Hubert feature for iteration 1."
    ./local/km.sh \
        --stage 1 --stop-stage 3 \
        --ncluster ${hubert_n_clusters} \
        --feature-type ${feature_hubert_iter1} \
        --datadir "./data" \
        --kmrootdir "./exp"
        --dictdir "./data/${feature_hubert_iter1}_km${hubert_n_clusters}_token_list/word"
fi

train_set="train_960_${feature_hubert_iter1}_km${hubert_n_clusters}"
valid_set="dev_clean_${feature_hubert_iter1}_km${hubert_n_clusters}"
test_sets="test_clean_${feature_hubert_iter1}_km${hubert_n_clusters} \
    test_other_${feature_hubert_iter1}_km${hubert_n_clusters} \
    dev_clean_${feature_hubert_iter1}_km${hubert_n_clusters} \
    dev_other_${feature_hubert_iter1}_km${hubert_n_clusters}"
#TODO(use asr.sh stage 4 to dump data)
# mkdir -p dump/${feature_hubert_iter1}_km${hubert_n_clusters}/raw
# for task in ${train_set} ${valid_set} ${test_sets}; do
#     cp -r data/${task} dump/${feature_hubert_iter1}_km${hubert_n_clusters}/raw
# done

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Pretrain HuBERT model using ${feature_hubert_iter1} K-means pseudo-labels."

    ./hubert_asr.sh \
        --stage 9 --stop-stage 10 \
        --lang ${feature_hubert_iter1}_km${hubert_n_clusters} \
        --ngpu ${ngpu} \
        --nj 32 \
        --max_wav_duration 30 \
        --asr_config "${pretrain_iter1_config}" \
        --dumpdir "dump/${feature_hubert_iter1}_km${hubert_n_clusters}" \
        --train-set "${train_set}" \
        --valid-set "${valid_set}" \
        --test_sets "${test_sets}" \
        --feats-normalize null \
        --token_type word "$@"
fi

