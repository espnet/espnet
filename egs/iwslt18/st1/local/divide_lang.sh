#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 tc data/dev"
    exit 1
fi

set=$1
# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for En
mkdir -p data/${set}.en
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.en/${f}
    fi
done
if [ ${set} = "train_nodevtest" ] || [ ${set} = "train_nodevtest_sp" ] || [ ${set} = "dev" ] || [ ${set} = "test" ]; then
    sort data/${set}/text.lc.rm.en > data/${set}.en/text  # dummy
    sort data/${set}/text.tc.en > data/${set}.en/text.tc
    sort data/${set}/text.lc.en > data/${set}.en/text.lc
    sort data/${set}/text.lc.rm.en > data/${set}.en/text.lc.rm
else
    sort data/${set}/text_noseg.tc.en > data/${set}.en/text_noseg.tc
    sort data/${set}/text_noseg.lc.en > data/${set}.en/text_noseg.lc
    sort data/${set}/text_noseg.lc.rm.en > data/${set}.en/text_noseg.lc.rm
fi
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.en
if [ ${set} = "train_nodevtest" ] || [ ${set} = "train_nodevtest_sp" ] || [ ${set} = "dev" ] || [ ${set} = "test" ]; then
    if [ -f data/${set}.en/feats.scp ]; then
        utils/validate_data_dir.sh data/${set}.en || exit 1;
    else
        utils/validate_data_dir.sh --no-feats data/${set}.en || exit 1;
    fi
else
    if [ -f data/${set}.en/feats.scp ]; then
        utils/validate_data_dir.sh --no-text data/${set}.en || exit 1;
    else
        utils/validate_data_dir.sh --no-text --no-feats data/${set}.en || exit 1;
    fi
fi

# for De
mkdir -p data/${set}.de
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.de/${f}
    fi
done
if [ ${set} = "train_nodevtest" ] || [ ${set} = "train_nodevtest_sp" ] || [ ${set} = "dev" ] || [ ${set} = "test" ]; then
    sort data/${set}/text.tc.de > data/${set}.de/text  # dummy
    sort data/${set}/text.tc.de > data/${set}.de/text.tc
    sort data/${set}/text.lc.de > data/${set}.de/text.lc
    sort data/${set}/text.lc.rm.de > data/${set}.de/text.lc.rm
else
    sort data/${set}/text_noseg.tc.de > data/${set}.de/text_noseg.tc
    sort data/${set}/text_noseg.lc.de > data/${set}.de/text_noseg.lc
    sort data/${set}/text_noseg.lc.rm.de > data/${set}.de/text_noseg.lc.rm
fi
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.de
if [ ${set} = "train_nodevtest" ] || [ ${set} = "train_nodevtest_sp" ] || [ ${set} = "dev" ] || [ ${set} = "test" ]; then
    if [ -f data/${set}.de/feats.scp ]; then
        utils/validate_data_dir.sh data/${set}.de || exit 1;
    else
        utils/validate_data_dir.sh --no-feats data/${set}.de || exit 1;
    fi
else
    if [ -f data/${set}.de/feats.scp ]; then
        utils/validate_data_dir.sh --no-text data/${set}.de || exit 1;
    else
        utils/validate_data_dir.sh --no-text --no-feats data/${set}.de || exit 1;
    fi
fi
