#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr05_multi_noisy_si284 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="\
dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
"

asr_config=conf/train_asr_e_branchformer.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm_transformer.yaml

speed_perturb_factors="0.9 1.0 1.1"

use_word_lm=false
word_vocab_size=65000

./asr.sh                                   \
    --lang en \
    --nlsyms_txt data/nlsyms.txt           \
    --token_type char                      \
    --feats_type raw                       \
    --audio_format flac.ark                \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --word_vocab_size ${word_vocab_size}   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" "$@"
