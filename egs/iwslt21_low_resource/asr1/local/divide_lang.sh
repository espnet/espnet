#!/usr/bin/env bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <set> <src_lang> <tgt_lang>"
    echo "e.g.: $0 dev"
    exit 1
fi

set=$1
src_lang=$2
tgt_lang=$3

# Copy stuff into its final locations [this has been moved from the format_data script]
# for ${src_lang}
mkdir -p data/${set}.${src_lang}
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.${src_lang}/${f}
    fi
done
sort data/${set}/text.lc.rm.${src_lang} > data/${set}.${src_lang}/text  # dummy
sort data/${set}/text.tc.${src_lang} > data/${set}.${src_lang}/text.tc
sort data/${set}/text.lc.${src_lang} > data/${set}.${src_lang}/text.lc
sort data/${set}/text.lc.rm.${src_lang} > data/${set}.${src_lang}/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.${src_lang}
if [ -f data/${set}.${src_lang}/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.${src_lang} || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav data/${set}.${src_lang} || exit 1;
fi

# for target language
mkdir -p data/${set}.${tgt_lang}
for f in spk2utt utt2spk segments wav.scp feats.scp utt2num_frames; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.${tgt_lang}/${f}
    fi
done
sort data/${set}/text.tc.${tgt_lang} > data/${set}.${tgt_lang}/text  # dummy
sort data/${set}/text.tc.${tgt_lang} > data/${set}.${tgt_lang}/text.tc
sort data/${set}/text.lc.${tgt_lang} > data/${set}.${tgt_lang}/text.lc
sort data/${set}/text.lc.rm.${tgt_lang} > data/${set}.${tgt_lang}/text.lc.rm
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.${tgt_lang}
if [ -f data/${set}.${tgt_lang}/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.${tgt_lang} || exit 1;
else
    utils/validate_data_dir.sh --no-feats --no-wav data/${set}.${tgt_lang} || exit 1;
fi
