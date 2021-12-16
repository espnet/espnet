#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"

test_sets=""
for l in ${recog}; do
  test_sets="dev_${l} eval_${l} ${test_sets}"
done
test_sets=${test_sets%% }

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

nlsyms_txt=data/nlsym.txt


# TODO(kamo): Derive language name from $langs and give it as --lang
./asr.sh \
    --lang noinfo \
    --local_data_opts "--langs ${langs} --recog ${recog}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type char \
    --feats_type raw \
    --nlsyms_txt ${nlsyms_txt} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
