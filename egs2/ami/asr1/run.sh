#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm

train_set=${mic}_train
valid_set=${mic}_dev
test_sets="${mic}_eval ${mic}_dev"

asr_config=conf/tuning/train_asr_transformer4.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/tuning/decode_transformer2.yaml

speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --lang en \
    --local_data_opts "--mic ${mic}" \
    --nbpe 100 \
    --use_lm true \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
