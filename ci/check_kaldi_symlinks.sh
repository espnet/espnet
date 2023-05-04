#!/usr/bin/env bash

# NOTE: DO NOT WRITE DISTRIBUTION-SPECIFIC COMMANDS HERE (e.g., apt, dnf, etc)

set -euo pipefail

(
    set -euo pipefail
    cd tools
    if [ ! -e kaldi ]; then
        git clone --depth 1 https://github.com/kaldi-asr/kaldi
    fi
)

while IFS= read -r f; do
    if [ "${f}" = utils/README.md ]; then
        continue
    fi
    if [ "${f}" = steps/README.md ]; then
        continue
    fi

    echo diff egs2/TEMPLATE/asr1/$f tools/kaldi/egs/wsj/s5/$f
    diff egs2/TEMPLATE/asr1/$f tools/kaldi/egs/wsj/s5/$f
done< <(cd egs2/TEMPLATE/asr1; find utils steps -type f)



while IFS= read -r f; do
    if [ "${f}" = sid/README.md ]; then
        continue
    fi

    echo diff egs2/TEMPLATE/tts1/$f tools/kaldi/egs/sre08/v1/$f
    diff egs2/TEMPLATE/tts1/$f tools/kaldi/egs/sre08/v1/$f
done< <(cd egs2/TEMPLATE/tts1; find sid -type f)
