#!/bin/bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"

train_test=""
for l in ${recog}; do
  train_test="eval_${l} ${train_test}"
done
train_test=${train_test%% }

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
decode_config=conf/decode_asr.yaml

nlsyms_txt=data/nlsym.txt

./asr.sh \
    --local_data_opts "--langs ${langs} --recog ${recog}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type char \
    --feats_type raw \
    --nlsyms_txt ${nlsyms_txt} \
    --asr_config "${asr_config}" \
    --decode_config "${decode_config}" \
    --train_set "${train_set}" \
    --dev_set "${train_dev}" \
    --eval_sets "${train_test}" \
    --srctexts "data/${train_set}/text" "$@"
