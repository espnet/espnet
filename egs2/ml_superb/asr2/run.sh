#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Process Pipeline

stage=1
stop_stage=100
nj=32
inference_nj=4
gpu_inference=true

# Config
duration=10min
multilingual=true
lid=false
only_lid=false
single_lang=xty

# Model/Inference Configs
inference_config=conf/decode.yaml
asr_config=conf/train.yaml

./utils/parse_options.sh || exit 1

# Common configs for ML-SUPERB
tgt_token_type=char
if "${multilingual}"; then
    if "${only_lid}"; then
        suffix="_only_lid"
        tgt_token_type=word
    else
        if "${lid}"; then
            suffix="_lid"
        else
            suffix=""
        fi
    fi
    train_set=train_${duration}${suffix}
    train_dev=dev_${duration}${suffix}
    test_set="${train_dev} test_${duration}${suffix}"
    lang="multilingual"
else
    train_set=train_${duration}_${single_lang}
    train_dev=dev_${duration}_${single_lang}
    test_set="${train_dev} test_${duration}_${single_lang}"
    lang=${single_lang}
fi

nlsyms_txt=data/local/nlsyms.txt
asr_tag="$(basename "${asr_config}" .yaml)_${lang}_${duration}"

local_data_opts="--duration ${duration} --lid ${lid} --only_lid ${only_lid}"
local_data_opts+=" --multilingual ${multilingual} --single_lang ${single_lang} --nlsyms_txt ${nlsyms_txt}"

kmeans_feature="wavlm_large/21"  # use model_type/layer_index
nclusters=2000

src_lang=$(echo "${kmeans_feature}_km${nclusters}" | tr "/" "_")
tgt_lang=${lang}

src_nbpe=3000   # I use src_nbpe=6000 for 2000-cluster kmeans.
tgt_nbpe=6000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./asr2.sh \
    --ngpu 1 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj ${nj} \
    --inference_nj ${inference_nj} \
    --gpu_inference ${gpu_inference} \
    --lang ${lang} \
    --inference_asr_model valid.acc.ave.pth \
    --local_data_opts "${local_data_opts}" \
    --nlsyms_txt ${nlsyms_txt} \
    --use_lm false \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --asr_tag "${asr_tag}" \
    --asr_stats_dir exp/asr_stats_${lang}_${duration} \
    --kmeans_opts "--batch_bins 3600000" \
    --kmeans_feature "${kmeans_feature}" \
    --nclusters "${nclusters}" \
    --portion 0.3 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type ${tgt_token_type} \
    --tgt_nbpe $tgt_nbpe \
    --src_case "${src_case}" \
    --tgt_case "${tgt_case}" \
    --tgt_case "${tgt_case}" \
    --audio_format "flac.ark" \
    --src_bpe_train_text "dump/raw/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "dump/raw/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "dump/raw/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --local_score_opts "${lid} ${only_lid} normal"
