#!/usr/bin/env bash
set -euo pipefail

# Build the TEDxJP-10K dataset

. ./path.sh
set -e # exit on error

. utils/parse_options.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <raw_data_dir>"
  echo "This script does preprocessing of the TEDx-JP-10K dataset."
  echo "<raw_data_dir> should contain segments, spk2utt, text, utt2spk, wavlist.txt, and wav/."
  exit 1
fi

RAW_DATA_DIR=$1
dst_dir="data/tedx-jp-10k"

mkdir -p ${dst_dir}

# Copy necessary files to data directory
echo "$0: Copying segments, spk2utt, text and utt2spk to $dst_dir."
cp ${RAW_DATA_DIR}/{segments,spk2utt,text,utt2spk} ${dst_dir}

echo "$0: Creating wav.scp from wavlist.txt"
rm -f ${dst_dir}/wav.scp
touch ${dst_dir}/wav.scp
while read line; do
    id=$(cut -d' ' -f 1 <<<${line})
    filepath=${RAW_DATA_DIR}/wav/$(cut -d' ' -f 2 <<<${line})
    echo "${id} sox \"${filepath}\" -c 1 -r 16000 -t wav - |" >> ${dst_dir}/wav.scp
done < ${RAW_DATA_DIR}/wavlist.txt
utils/data/validate_data_dir.sh --no-feats ${dst_dir}

echo "$0: Done preprocessing TEDxJP-10K dataset (${dst_dir})"
