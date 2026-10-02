#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/tuning/train_asr_xeus.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# Both configs put a frozen SSL model in front of the encoder, and the
# global_mvn statistics asr.sh collects are log-mel statistics -- meaningless
# for those features.  See espnet/espnet#4006.  The config is read rather than
# matched on its name, so renaming one cannot silently normalise SSL features
# with log-mel statistics; asr_config may also arrive on the command line, which
# is appended after these defaults, so look there too.
_asr_config="${asr_config}"
_prev=
for _arg in "$@"; do
    [ "${_prev}" = --asr_config ] && _asr_config="${_arg}"
    _prev="${_arg}"
done
if grep -qE '^frontend:[[:space:]]*(s3prl|espnet_ssl|huggingface)' "${_asr_config}"; then
    feats_normalize=utt_mvn
else
    feats_normalize=global_mvn
fi

# Use "--local_data_opts '--tier utt.prono.'" to train on the pronounced
# (colloquial) transcription instead of the orthographic one.
local_data_opts=""

./asr.sh \
    --lang ko \
    --audio_format flac.ark \
    --nbpe 2000 \
    --bpe_nlsyms data/nlsyms.txt \
    --nlsyms_txt data/nlsyms.txt \
    --use_lm false \
    --feats_normalize "${feats_normalize}" \
    --min_wav_duration 1.0 \
    --max_wav_duration 30 \
    --local_data_opts "${local_data_opts}" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"
