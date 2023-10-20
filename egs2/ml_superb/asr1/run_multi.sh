#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Process Pipeline
stage=1
stop_stage=13
nj=32
inference_nj=4
gpu_inference=true

# Config
duration=10min # duration set ("10min" or "1h")
only_lid=false # whether to perform only the LID task
lid=false # whether to add joint LID task in multiligual ASR

# Model/Inference Configs
inference_config=conf/decode_asr.yaml
asr_config=conf/tuning/train_asr_fbank_${duration}.yaml

. utils/parse_options.sh || exit 1

# Common configs for ML-SUPERB
token_type=char
if "${only_lid}"; then
    suffix="_only_lid"
    token_type=word
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

nlsyms_txt=data/local/nlsyms.txt
asr_tag="$(basename "${asr_config}" .yaml)_multilingual_${duration}"

local_data_opts="--duration ${duration} --lid ${lid} --only_lid ${only_lid}"
local_data_opts+=" --multilingual true --nlsyms_txt ${nlsyms_txt}"

./asr.sh \
    --ngpu 1 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj ${nj} \
    --inference_nj ${inference_nj} \
    --gpu_inference ${gpu_inference} \
    --lang "multilingual_${duration}_${suffix}" \
    --inference_asr_model valid.loss.ave.pth \
    --local_data_opts "${local_data_opts}" \
    --nlsyms_txt ${nlsyms_txt} \
    --use_lm false \
    --token_type ${token_type} \
    --feats_type raw \
    --feats_normalize utterance_mvn \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --asr_tag "${asr_tag}" \
    --asr_stats_dir exp/asr_stats_multilingual_${duration} \
    --local_score_opts "${lid} ${only_lid} normal"
