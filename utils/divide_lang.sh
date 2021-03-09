#!/bin/bash

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <set> <langs divided by space>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1
langs=$2

# Copy stuff intoc its final locations [this has been moved from the format_data script]
for lang in ${langs}; do
    mkdir -p data/${set}.${lang}
    for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
        if [ -f data/${set}/${f} ]; then
            sort data/${set}/${f} > data/${set}.${lang}/${f}
        fi
    done
    sort data/${set}/text.lc.rm.${lang} > data/${set}.${lang}/text  # dummy
    for case in lc.rm lc tc; do
        sort data/${set}/text.${case}.${lang} > data/${set}.${lang}/text.${case}
    done
    utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.${lang}
    if [ -f data/${set}.${lang}/feats.scp ]; then
        utils/validate_data_dir.sh data/${set}.${lang} || exit 1;
    else
        utils/validate_data_dir.sh --no-feats --no-wav data/${set}.${lang} || exit 1;
    fi
done
