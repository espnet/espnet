#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <set>"
    echo "e.g.: $0 fisher_dev"
    exit 1
fi

set=$1

function check_sorted {
    file=$1
    sort -k1,1 -u <${file} >${file}.tmp
    if ! cmp -s ${file} ${file}.tmp; then
        echo "$0: file $1 is not in sorted order or not unique, sorting it"
        mv ${file}.tmp ${file}
    else
        rm ${file}.tmp
    fi
}

# fix the original directory at first
cp -rf data/${set} data/${set}.tmp
cut -f 1 -d " " data/${set}/utt2spk > data/${set}/reclist
reduce_data_dir.sh data/${set}.tmp data/${set}/reclist data/${set}
if [ -f data/${set}/text.en ]; then
    utils/fix_data_dir.sh --utt_extra_files "text.tc.es text.tc.en text.lc.es text.lc.en text.lc.rm.es text.lc.rm.en" data/${set}
else
    utils/fix_data_dir.sh --utt_extra_files "text.tc.es text.tc.en.0 text.tc.en.1 text.tc.en.2 text.tc.en.3 \
                                             text.lc.es text.lc.en.0 text.lc.en.1 text.lc.en.2 text.lc.en.3 \
                                             text.lc.rm.es text.lc.rm.en.0 text.lc.rm.en.1 text.lc.rm.en.2 text.lc.rm.en.3" data/${set}
fi
rm -rf data/${set}.tmp


# Copy stuff intoc its final locations [this has been moved from the format_data script]
# for Es
mkdir -p data/${set}.es
for f in spk2utt utt2spk segments feats.scp wav.scp utt2num_frames spk2gender reco2file_and_channel; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.es/${f}
    fi
done
sort data/${set}/text.lc.rm.es > data/${set}.es/text && check_sorted data/${set}.es/text; # dummy
sort data/${set}/text.tc.es > data/${set}.es/text.tc && check_sorted data/${set}.es/text.tc;
sort data/${set}/text.lc.es > data/${set}.es/text.lc && check_sorted data/${set}.es/text.lc;
sort data/${set}/text.lc.rm.es > data/${set}.es/text.lc.rm && check_sorted data/${set}.es/text.lc.rm;
utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.es || exit 1;
if [ -f data/${set}.es/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.es || exit 1;
else
    utils/validate_data_dir.sh --no-feats data/${set}.es || exit 1;
fi

# for En
mkdir -p data/${set}.en
for f in spk2utt utt2spk segments feats.scp wav.scp utt2num_frames spk2gender reco2file_and_channel; do
    if [ -f data/${set}/${f} ]; then
        sort data/${set}/${f} > data/${set}.en/${f}
    fi
done
if [ -f data/${set}/text.tc.en ]; then
    sort data/${set}/text.tc.en > data/${set}.en/text && check_sorted data/${set}.en/text;  # dummy
    sort data/${set}/text.tc.en > data/${set}.en/text.tc && check_sorted data/${set}.en/text.tc;
    sort data/${set}/text.lc.en > data/${set}.en/text.lc && check_sorted data/${set}.en/text.lc;
    sort data/${set}/text.lc.rm.en > data/${set}.en/text.lc.rm && check_sorted data/${set}.en/text.lc.rm;
    utils/fix_data_dir.sh --utt_extra_files "text.tc text.lc text.lc.rm" data/${set}.en || exit 1;
else
    sort data/${set}/text.tc.en.0 > data/${set}.en/text  # dummy
    sort data/${set}/text.tc.en.0 > data/${set}.en/text.tc
    sort data/${set}/text.lc.en.0 > data/${set}.en/text.lc
    sort data/${set}/text.lc.rm.en.0 > data/${set}.en/text.lc.rm
    for no in 1 2 3; do
        sort data/${set}/text.tc.en.${no} > data/${set}.en/text.tc.${no}
        sort data/${set}/text.lc.en.${no} > data/${set}.en/text.lc.${no}
        sort data/${set}/text.lc.rm.en.${no} > data/${set}.en/text.lc.rm.${no}
    done
    utils/fix_data_dir.sh --utt_extra_files "text.tc text.tc.1 text.tc.2 text.tc.3 \
                                             text.lc text.lc.1 text.lc.2 text.lc.3 \
                                             text.lc.rm text.lc.rm.1 text.lc.rm.2 text.lc.rm.3" data/${set}.en || exit 1;
fi
if [ -f data/${set}.en/feats.scp ]; then
    utils/validate_data_dir.sh data/${set}.en || exit 1;
else
    utils/validate_data_dir.sh --no-feats data/${set}.en || exit 1;
fi
