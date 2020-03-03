#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
dev_set=dt05_multi_isolated_1ch_track
eval_set="\
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
"

asr_config=conf/train_asr_rnn.yaml
decode_config=conf/decode_asr_rnn.yaml
lm_config=conf/train_lm.yaml


use_word_lm=false
word_vocab_size=65000

./asr.sh                                   \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type char                      \
    --feats_type fbank_pitch               \
    --asr_config "${asr_config}"           \
    --decode_config "${decode_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --dev_set "${dev_set}"                 \
    --eval_sets "${eval_set}"              \
    --srctexts "data/${train_set}/text data/local/other_text/text" "$@"
