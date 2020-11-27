#!/usr/bin/env bash

# Data preparation for LaboroTVSpeech

. ./path.sh
set -e # exit on error

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <corpus_dir>"
  exit 1
fi

CORPUS_DIR=$1

# Data
for x in train dev; do
  echo "$0: Making data/${x} ..."
  mkdir -p data/${x}
  perl -pe 's/,/ /' ${CORPUS_DIR}/data/${x}/text.csv >data/${x}/text
  cut -d',' -f1 ${CORPUS_DIR}/data/${x}/text.csv |
    awk -v dir=${CORPUS_DIR}/data/${x}/wav/ "{print dir\$1\".wav\"}" |
    sort |
    perl -pe 's,(.*/)([^/]*)(\.wav),\2 \1\2\3,g' \
      >data/${x}/wav.scp
  # find -L ${CORPUS_DIR}/data/${x}/wav -name "*.wav" |
  #   sort |
  #   perl -pe 's,(.*/)([^/]*)(\.wav),\2 \1\2\3,g' \
  #     >data/${x}/wav.scp

  # Make a dumb utt2spk and spk2utt,
  # where each utterance corresponds to a unique speaker.
  awk '{print $1,$1_spk}' data/${x}/text >data/${x}/utt2spk
  utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk >data/${x}/spk2utt
  utils/fix_data_dir.sh data/${x}
  utils/validate_data_dir.sh --no-feats data/${x}
done

echo "$0: done preparing data directories"
