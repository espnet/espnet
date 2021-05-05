#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <src-dir>"
  echo "e.g.: $0 data/dev"
  exit 1
fi

src=$1
set=$(echo $src | awk -F"/" '{print $NF}')

# Copy stuff intoc its final locations [this has been moved from the format_data script]
for lang in en fr de; do
    mkdir -p data/$set.${lang}
    for f in spk2utt utt2spk wav.scp feats.scp utt2num_frames; do
        sort data/$set/$f > data/$set.${lang}/$f
    done
    sort data/$set/text.lc.${lang} > data/$set.${lang}/text

    utils/fix_data_dir.sh data/$set.${lang}
    utils/validate_data_dir.sh --no-feats data/$set.${lang} || exit 1;
done
