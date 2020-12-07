#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodev
valid_set=dev_4k
# NOTE: tedx-jp-10k_verbatim generation script has some error.
#   See https://github.com/laboroai/LaboroTVSpeech/issues/1
#   If you faced on this issue, you can only use the `dev_4k` and `dev` for
#   the test sets by changing to `test_sets="dev_4k dev"` and adding the
#   option `--local_data_opts "--stage 3"`.
test_sets="dev_4k dev tedx-jp-10k_verbatim"

asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --ngpu 4 \
    --nj 128 \
    --inference_nj 256 \
    --lang jp \
    --token_type char \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"
