#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1

# Copy stuff into its final locations [this has been moved from the format_data script]
mkdir -p data/${set}.es
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.es/${f}
    fi
done
sort data/${set}/text.tc.es > data/${set}.es/text  # dummy
sort data/${set}/text.tc.es > data/${set}.es/text.tc
utils/fix_data_dir.sh --utt_extra_files "text.tc" data/${set}.es
if [ -f data/${set}.es/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.es || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav data/${set}.es || exit 1;
fi

# for target language
mkdir -p data/${set}.na
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.na/${f}
    fi
done
sort data/${set}/text.tc.na > data/${set}.na/text  # dummy
sort data/${set}/text.tc.na > data/${set}.na/text.tc
utils/fix_data_dir.sh --utt_extra_files "text.tc" data/${set}.na
if [ -f data/${set}.na/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.na || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav data/${set}.na || exit 1;
fi
