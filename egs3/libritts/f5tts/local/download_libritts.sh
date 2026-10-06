#!/usr/bin/env bash

# Download and extract LibriTTS dataset from OpenSLR
# Adapted from egs2/libritts/tts1/local/download_and_untar.sh
# Usage: download_libritts.sh <data-base> <corpus-part>
# e.g.: download_libritts.sh /export/data dev-clean

set -e
set -o pipefail

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <corpus-part>"
  echo "e.g.: $0 /export/data libritts/data dev-clean"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: dev-clean, test-clean, dev-other, test-other,"
  echo "          train-clean-100, train-clean-360, train-other-500."
  exit 1
fi

data=$1
part=$2

url_base=www.openslr.org/resources/60

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1
fi

part_ok=false
list="dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500"
for x in $list; do
  if [ "$part" == "$x" ]; then part_ok=true; fi
done
if ! $part_ok; then
  echo "$0: expected <corpus-part> to be one of $list, but got '$part'"
  exit 1
fi

if [ -f "$data/LibriTTS/$part/.complete" ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0
fi

# Sizes of the archive files in bytes (for validation)
sizes="1291469655 924804676 1230670113 964502297 7723686890 27504073644 44565031479"

if [ -f "$data/$part.tar.gz" ]; then
  size=$(/bin/ls -l "$data/$part.tar.gz" | awk '{print $5}')
  size_ok=false
  for s in $sizes; do if [ "$s" == "$size" ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $data/$part.tar.gz because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm "$data/$part.tar.gz"
  else
    echo "$data/$part.tar.gz exists and appears to be complete."
  fi
fi

if [ ! -f "$data/$part.tar.gz" ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1
  fi
  full_url=$url_base/$part.tar.gz
  echo "$0: downloading data from $full_url. This may take some time, please be patient."

  if ! wget -P "$data" --no-check-certificate "$full_url"; then
    echo "$0: error executing wget $full_url"
    exit 1
  fi
fi

mkdir -p "$data/LibriTTS"

if ! tar -C "$data" -xzf "$data/$part.tar.gz"; then
  echo "$0: error un-tarring archive $data/$part.tar.gz"
  exit 1
fi

mkdir -p "$data/LibriTTS/$part"
touch "$data/LibriTTS/$part/.complete"

echo "$0: Successfully downloaded and un-tarred $part"

if $remove_archive; then
  echo "$0: removing $data/$part.tar.gz file since --remove-archive option was supplied."
  rm "$data/$part.tar.gz"
fi
