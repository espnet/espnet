#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1

# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p data/${set}.en
for f in spk2utt utt2spk segments feats.scp; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.en/${f}
    fi
done
sort data/${set}/text.lc.rm.en > data/${set}.en/text  # dummy
sort data/${set}/text.tc.en > data/${set}.en/text.tc
sort data/${set}/text.lc.en > data/${set}.en/text.lc
sort data/${set}/text.lc.rm.en > data/${set}.en/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.en
utils/validate_data_dir.sh --no-wav data/${set}.en || exit 1;

# for Pt
mkdir -p data/${set}.pt
for f in spk2utt utt2spk segments feats.scp; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.pt/${f}
    fi
done
sort data/${set}/text.tc.pt > data/${set}.pt/text  # dummy
sort data/${set}/text.tc.pt > data/${set}.pt/text.tc
sort data/${set}/text.lc.pt > data/${set}.pt/text.lc
sort data/${set}/text.lc.rm.pt > data/${set}.pt/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.pt
utils/validate_data_dir.sh --no-wav data/${set}.pt || exit 1;
