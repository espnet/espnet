#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050
n_shift=256

# For TTS, length and stress marks should be grouped together with the phoneme
# Also, we need the word separator to identify words for G2P dictionary
# >>> espeak_ng_english_us_vits('cool, right??')
# >>> ['k', 'ˈ', 'u', 'ː', 'l' ',', '<space>', 'ɹ', 'ˈ', 'a', 'ɪ', 't', '?', '?']
# >>> espeak_ng_english_us_word_sep('cool, right??')
# >>> ['k', 'ˈuː', 'l', '|', ',', 'ɹ', 'ˈaɪ', 't', '|', '??']

./scripts/utils/mfa.sh \
    --language english_us_espeak  \
    --train true \
    --cleaner mfa_english \
    --g2p_model espeak_ng_english_us_word_sep \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --clean_temp true \
    --single_speaker true \
    "$@"
