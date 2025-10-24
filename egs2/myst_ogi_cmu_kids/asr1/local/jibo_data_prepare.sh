#!/bin/bash

wav_dir=$1
data=$2

tmpdir=./tmp
mkdir -p $tmpdir

for x in all; do
  [ ! -d $data/$x ] && mkdir -p $data/$x

  # Find all .wav files
  find $wav_dir -iname "*.wav" > $tmpdir/all.wav

  # Find all transcription files
  find $wav_dir -iname "*.txt" ! -iname "README.txt" > $tmpdir/all.txt

  # Generate utt2spk
  while read -r trans_file; do
    category=$(echo "$trans_file" | sed -E 's:.*/data/([^/]+)/.*:\1:')
    base=$(basename "$trans_file" .txt)
    spk=$(echo "$base" | sed -E 's:.*([0-9]{3})$:\1:')

    awk -v category="$category" -v base="$base" -v spk="$spk" '{
      printf("%s_%s-%07d-%07d %s\n", spk "_" base, category, int($1 * 100), int($2 * 100), spk)
    }' "$trans_file"
  done < $tmpdir/all.txt | sort -k1,1 -u > $data/$x/utt2spk

  # Generate wav.scp
  sed -E 's:(.*/data/([^/]+)/((T[0-9]+)([0-9]{3})))/.*\.wav$:\5_\3_\2 \1/\3.wav:' $tmpdir/all.wav | sort -k1,1 > $data/$x/wav.scp

  # Generate text file
  while read -r trans_file; do
    category=$(echo "$trans_file" | sed -E 's:.*/data/([^/]+)/.*:\1:')
    base=$(basename "$trans_file" .txt)
    spk=$(echo "$base" | sed -E 's:.*([0-9]{3})$:\1:')

    awk -v category="$category" -v base="$base" -v spk="$spk" '{
      printf("%s_%s-%07d-%07d %s\n", spk "_" base, category, int($1 * 100), int($2 * 100), $3)
    }' "$trans_file"
  done < $tmpdir/all.txt | sort -k1,1 -u > $data/$x/text

  # Check consistency
  ntrans=$(wc -l < $data/$x/text)
  nutt2spk=$(wc -l < $data/$x/utt2spk)
  if [ "$ntrans" -ne "$nutt2spk" ]; then
    echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1
  fi

  # Generate segments
  while read -r trans_file; do
    category=$(echo "$trans_file" | sed -E 's:.*/data/([^/]+)/.*:\1:')
    base=$(basename "$trans_file" .txt)
    spk=$(echo "$base" | sed -E 's:.*([0-9]{3})$:\1:')

    awk -v category="$category" -v base="$base" -v spk="$spk" '{
      if ($1 != $2) {  # Only include non-zero-duration segments
        printf("%s_%s-%07d-%07d %s_%s %.3f %.3f\n", spk "_" base, category, int($1 * 100), int($2 * 100), spk "_" base, category, $1, $2)
      }
    }' "$trans_file"
  done < $tmpdir/all.txt | sort -k1,1 -u > $data/$x/segments

  # Filter files for valid segments
  cut -d' ' -f1 $data/$x/segments > $tmpdir/valid_utts

  # Filter utt2spk
  grep -Ff $tmpdir/valid_utts $data/$x/utt2spk > $data/$x/utt2spk.filtered
  mv $data/$x/utt2spk.filtered $data/$x/utt2spk

  # Filter text
  grep -Ff $tmpdir/valid_utts $data/$x/text > $data/$x/text.filtered
  mv $data/$x/text.filtered $data/$x/text

  # Generate spk2utt
  utils/utt2spk_to_spk2utt.pl < $data/$x/utt2spk > $data/$x/spk2utt || exit 1

  # Validate directory
  utils/validate_data_dir.sh --no-feats $data/$x || exit 1
done

# Generate non-segmented data
x_no_segments="all_no_segments"
[ ! -d $data/$x_no_segments ] && mkdir -p $data/$x_no_segments

# Copy wav.scp
cp $data/$x/wav.scp $data/$x_no_segments/wav.scp

# Generate utt2spk for non-segmented data
awk '{split($1, a, "_"); print $1, a[1]}' $data/$x_no_segments/wav.scp | sort -k1,1 > $data/$x_no_segments/utt2spk

# Generate text for non-segmented data
while read -r trans_file; do
  category=$(echo "$trans_file" | sed -E 's:.*/data/([^/]+)/.*:\1:')
  base=$(basename "$trans_file" .txt)
  spk=$(echo "$base" | sed -E 's:.*([0-9]{3})$:\1:')

  awk -v category="$category" -v base="$base" -v spk="$spk" '{
    if (NR == 1) { transcript = $3 }
    else { transcript = transcript " " $3 }
  } END {
    printf("%s_%s_%s %s\n", spk, base, category, transcript)
  }' "$trans_file"
done < $tmpdir/all.txt | sort -k1,1 > $data/$x_no_segments/text

# Generate spk2utt for non-segmented data
utils/utt2spk_to_spk2utt.pl < $data/$x_no_segments/utt2spk > $data/$x_no_segments/spk2utt || exit 1

# Validate non-segmented directory
utils/validate_data_dir.sh --no-feats $data/$x_no_segments || exit 1

# Clean up temporary files
rm -r $tmpdir

echo "$0: successfully prepared data in $data"
exit 0
