#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1

# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p data/${set}.en
for f in spk2utt utt2spk wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.en/${f}
    fi
done
sort data/${set}/text.lc.rm.en > data/${set}.en/text  # dummy
sort data/${set}/text.tc.en > data/${set}.en/text.tc
sort data/${set}/text.lc.en > data/${set}.en/text.lc
sort data/${set}/text.lc.rm.en > data/${set}.en/text.lc.rm

utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.en
utils/validate_data_dir.sh --no-feats data/${set}.en || exit 1;

# for Fr
for lang in fr fr.gtranslate; do
    mkdir -p data/${set}.${lang}
    for f in spk2utt utt2spk wav.scp feats.scp utt2num_frames; do
        if [ -f data/${set}/${f} ]; then
            sort data/${set}/${f} > data/${set}.${lang}/${f}
        fi
    done
    sort data/${set}/text.tc.${lang} > data/${set}.${lang}/text  # dummy
    sort data/${set}/text.tc.${lang} > data/${set}.${lang}/text.tc
    sort data/${set}/text.lc.${lang} > data/${set}.${lang}/text.lc
    sort data/${set}/text.lc.rm.${lang} > data/${set}.${lang}/text.lc.rm

    utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.${lang}
    utils/validate_data_dir.sh --no-feats data/${set}.${lang} || exit 1;
done
