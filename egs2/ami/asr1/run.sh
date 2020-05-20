#!/bin/bash
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
train_dev=${mic}_dev
train_test=${mic}_eval

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
decode_config=conf/decode_asr.yaml

speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --local_data_opts "--mic ${mic}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --use_word_lm false \
    --word_vocab_size 20000 \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${train_dev}" \
    --eval_sets "${train_test}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --srctexts "data/${train_set}/text" "$@"
