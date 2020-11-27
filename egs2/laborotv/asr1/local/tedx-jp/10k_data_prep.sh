#!/usr/bin/env bash
set -euo pipefail

# Build the TEDxJP-10K dataset

. ./path.sh
set -e # exit on error

videos_csv="local/tedx-jp/tedx-jp-10k.csv"
all_to_10k_utt_map="local/tedx-jp/all_to_10k_utt_map.txt"

. utils/parse_options.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <raw_data_dir>"
  echo "This script builds the TEDx-JP-10K dataset."
  echo "You have to put all required .wav and .ja.vtt files in <raw_data_dir>"
  echo "before running this script."
  exit 1
fi

NUM_RAW_UTTS_EXPECTED=59617
RAW_DATA_DIR=$1

dst_dir_all_raw="data/tedx-jp-all_raw"
tmp_dir_10k_raw="data/local/tedx-jp-10k_raw.tmp"
tmp_dir_10k_verbatim="data/local/tedx-jp-10k_verbatim.tmp"
dst_dir_10k_verbatim="data/tedx-jp-10k_verbatim"

# Make sure all required files exist
cut -d ',' -f2 ${videos_csv} | tail -n+2 | while read -r video_id; do
  for file in "${RAW_DATA_DIR}/${video_id}.ja.vtt" "${RAW_DATA_DIR}/${video_id}.wav"; do
    if [ ! -e ${file} ]; then
      echo "$0: ${file} does not exist. "
      echo "    Please prepare all the files before executing this script."
    fi
  done
done
echo "$0: All required files exist in ${RAW_DATA_DIR}."

# Firstly, make a kaldi-format directory containing ALL data.
echo "$0: Creating a directory containing all valid subtitle segments."
python local/tedx-jp/make_raw_dir.py \
  --video-csv ${videos_csv} \
  ${RAW_DATA_DIR} ${dst_dir_all_raw}

num_raw_utts=$(wc -l <${dst_dir_all_raw}/utt2spk)
if [ ${num_raw_utts} -ne ${NUM_RAW_UTTS_EXPECTED} ]; then
  echo "$0: The number of utterances in ${dst_dir_all_raw} is ${num_raw_utts},"
  echo "    but we expect to have ${NUM_RAW_UTTS_EXPECTED} utterances."
  echo "    In case you cannot solve this problem, please contact the authors,"
  echo "    because the original subtitle data may have changed."
  exit 1
fi

# Create a directory without segments, so we can use subsegment_data_dir.sh later.
utils/data/convert_data_dir_to_whole.sh \
  ${dst_dir_all_raw} ${dst_dir_all_raw}_whole

mkdir -p ${tmp_dir_10k_raw} ${tmp_dir_10k_verbatim}
cut -d' ' -f1 ${all_to_10k_utt_map} >${tmp_dir_10k_raw}/utt_list
cut -d' ' -f2 ${all_to_10k_utt_map} >${tmp_dir_10k_verbatim}/utt_list

# Build the dataset by making a subset and modifying its segments and text.
for file in text segments; do
  # Collect 10K utterances
  # ** We use local/filter_scp.py here to handle slightly buggy data in TEDxJP-10K,
  #    caused by the process of manually modifying the timestamps & text.
  python local/filter_scp.py \
    ${tmp_dir_10k_raw}/utt_list \
    ${dst_dir_all_raw}/${file} \
    >${tmp_dir_10k_raw}/${file}

  # Make a file without utterance ids
  cut -d' ' -f2- ${tmp_dir_10k_raw}/${file} \
    >${tmp_dir_10k_raw}/${file}_content

  # Apply patches to fix timestamps and raw text
  cp \
    ${tmp_dir_10k_raw}/${file}_content \
    ${tmp_dir_10k_verbatim}
  patch \
    ${tmp_dir_10k_verbatim}/${file}_content <local/tedx-jp/${file}.patch

  # Create file with utt_ids
  paste \
    -d ' ' \
    ${tmp_dir_10k_verbatim}/utt_list \
    ${tmp_dir_10k_verbatim}/${file}_content \
    >${tmp_dir_10k_verbatim}/${file}
done

# Subset datadir
utils/data/subsegment_data_dir.sh \
  ${dst_dir_all_raw}_whole \
  ${tmp_dir_10k_verbatim}/segments \
  ${tmp_dir_10k_verbatim}/text \
  ${dst_dir_10k_verbatim}

utils/data/validate_data_dir.sh --no-feats ${dst_dir_10k_verbatim}

rm -r ${tmp_dir_10k_raw} ${tmp_dir_10k_verbatim}

echo "$0: Done building TEDxJP-10K dataset (${dst_dir_10k_verbatim})"
