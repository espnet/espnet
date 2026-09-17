#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# An SSL upstream replaces the log-mel frontend, and the global_mvn statistics
# asr.sh collects are log-mel statistics -- meaningless for those features.
# See https://github.com/espnet/espnet/issues/4006#issuecomment-1047898558
# asr_config may also arrive on the command line, which is appended after these
# defaults, so pick it up from there too rather than silently normalising SSL
# features with log-mel statistics.
_asr_config="${asr_config}"
_prev=
for _arg in "$@"; do
    [ "${_prev}" = --asr_config ] && _asr_config="${_arg}"
    _prev="${_arg}"
done
feats_normalize=global_mvn
case "${_asr_config}" in
    *wavlm*|*hubert*|*wav2vec*|*xlsr*|*xeus*) feats_normalize=utt_mvn ;;
esac

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
