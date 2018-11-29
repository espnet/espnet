#!/bin/bash
#

if [ -f ./path.sh ]; then . ./path.sh; fi

mkdir -p data/train data/dev

# Copy stuff into its final locations...

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/train/$f data/train/$f || exit 1;
done

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/dev/$f data/dev/$f || exit 1;
done

echo hkust_format_data succeeded.
