#!/usr/bin/env bash

set -e
set -u 
set -o pipefail

text=""
lang_dir=""
g2p="g2p_en"
oov="<unk>"
reduce_vocab=true

. utils/parse_options.sh || exit 1;
. ./path.sh

cut -d ' ' -f2- "${text}" | sed "s/ /\n/g" | sort -u > "${lang_dir}/lexicon.words"

python3 -m espnet2.bin.tokenize_text \
    --token_type phn \
    --input "${lang_dir}/lexicon.words" \
    --output "${lang_dir}/lexicon.phones" \
    --g2p "${g2p}" \
    --write_vocabulary false \
    --cutoff 0

if ${reduce_vocab}; then
    cat "${lang_dir}/lexicon.phones" | sed "s/[0-9]//g" | sed "s/'//g" | \
        paste -d " " "${lang_dir}/lexicon.words" - > "${lang_dir}/lexicon.txt"
else
    cat "${lang_dir}/lexicon.phones" | \
        paste -d " " "${lang_dir}/lexicon.words" - > "${lang_dir}/lexicon.txt"
fi
echo "${oov} ${oov}" >> "${lang_dir}/lexicon.txt"
