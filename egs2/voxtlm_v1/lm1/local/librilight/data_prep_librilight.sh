#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

# all utterances are FLAC compressed
if ! which flac >&/dev/null; then
   echo "Please install 'flac' on ALL worker nodes!"
   exit 1
fi

mkdir -p $dst || exit 1

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort | grep -v logdir); do
  reader=$(basename $reader_dir)
  if ! [ $reader -eq $reader ]; then  # not integer.
    echo "$0: unexpected subdirectory name $reader"
    exit 1
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1
    fi

    flists=$(find -L $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac)

    for f in ${flists}; do
      echo "${reader}-${chapter}-${f} ${chapter_dir}/${f}.flac"
    done >>${wav_scp}

    # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
    #       to be a different speaker. This is done for simplicity and because we want
    #       e.g. the CMVN to be calculated per-chapter
    for f in ${flists}; do
      echo "${reader}-${chapter}-${f} ${reader}-${chapter}"
    done >>${utt2spk}

  done
done

cat ${utt2spk} | sort -k 2 > ${utt2spk}.sorted && mv ${utt2spk}.sorted ${utt2spk}
awk '(ARGIND==1){wav_scp[$1]=$0} (ARGIND==2){print(wav_scp[$1])}' ${wav_scp} ${utt2spk} > ${wav_scp}.sorted && mv ${wav_scp}.sorted ${wav_scp}

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

nwav_scp=$(wc -l <$wav_scp)
nutt2spk=$(wc -l <$utt2spk)
! [ "$nwav_scp" -eq "$nutt2spk" ] && \
  echo "Inconsistent #wav_scp($nwav_scp) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats --no-text $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
