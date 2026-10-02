#!/usr/bin/env bash

# Download and extract LibriSpeech (OpenSLR resource 12) from OpenSLR.
# LibriSpeech is a DIFFERENT corpus from LibriTTS (OpenSLR resource 60): the
# LibriSpeech-PC cross-sentence eval reads 16 kHz flacs laid out as
# <root>/<spk>/<chap>/<utt>.flac, which only LibriSpeech provides.
# Adapted from local/download_libritts.sh.
# Usage: download_librispeech.sh <data-base> <corpus-part>
# e.g.: download_librispeech.sh /export/data test-clean

set -e
set -o pipefail

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <corpus-part>"
  echo "e.g.: $0 /export/data test-clean"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
  echo "<corpus-part> can be one of: dev-clean, test-clean, dev-other, test-other,"
  echo "          train-clean-100, train-clean-360, train-other-500."
  exit 1
fi

data=$1
part=$2

url_base=www.openslr.org/resources/12

# The LibriTTS archives land in the same directory and LibriTTS ships subsets
# with identical names (e.g. test-clean.tar.gz), so namespace the archive by
# corpus. Sharing the filename would make each script delete the other's
# tarball on every size check.
archive=$data/LibriSpeech_$part.tar.gz

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

if [ -f "$data/LibriSpeech/$part/.complete" ]; then
  echo "$0: data part $part was already successfully extracted, nothing to do."
  exit 0
fi

# Sizes of the archive files in bytes (for validation), taken from
# egs2/librispeech/asr1/local/download_and_untar.sh. The first list is the
# older release, the second the final one.
sizes_old="371012589 347390293 379743611 361838298 6420417880 23082659865 30626749128"
sizes_new="337926286 314305928 695964615 297279345 87960560420 33373768 346663984 328757843 6387309499 23049477885 30593501606"

if [ -f "$archive" ]; then
  size=$(/bin/ls -l "$archive" | awk '{print $5}')
  size_ok=false
  for s in $sizes_old $sizes_new; do if [ "$s" == "$size" ]; then size_ok=true; fi; done
  if ! $size_ok; then
    echo "$0: removing existing file $archive because its size in bytes $size"
    echo "does not equal the size of one of the archives."
    rm "$archive"
  else
    echo "$archive exists and appears to be complete."
  fi
fi

if [ ! -f "$archive" ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1
  fi
  full_url=$url_base/$part.tar.gz
  echo "$0: downloading data from $full_url. This may take some time, please be patient."

  if ! wget -O "$archive" --no-check-certificate "$full_url"; then
    echo "$0: error executing wget $full_url"
    rm -f "$archive"
    exit 1
  fi
fi

mkdir -p "$data/LibriSpeech"

if ! tar -C "$data" -xzf "$archive"; then
  echo "$0: error un-tarring archive $archive"
  exit 1
fi

mkdir -p "$data/LibriSpeech/$part"
touch "$data/LibriSpeech/$part/.complete"

echo "$0: Successfully downloaded and un-tarred $part"

if $remove_archive; then
  echo "$0: removing $archive file since --remove-archive option was supplied."
  rm "$archive"
fi
