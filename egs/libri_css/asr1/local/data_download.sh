#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <corpus-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora/LibriCSS"
  exit 1
fi

corpus_dir=$1

set -e -o pipefail

# If data is not already present, then download and unzip
if [ ! -d $corpus_dir/for_release ]; then
    echo "Downloading and unpacking LibriCSS data."
    CWD=`pwd`
    mkdir -p $corpus_dir

    cd $corpus_dir

    tmp_dir=$(mktemp -d -p downloads)

    # Download the data. If the data has already been downloaded, it
    # does nothing. (See wget -c)
    wget -c --load-cookies ${tmp_dir}/cookies.txt \
      "https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
      --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
      'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l' \
      -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l" \
      -O for_release.zip
    rm -rf ${tmp_dir}

    # unzip (skip if already extracted)
    unzip -n for_release.zip

    # segmentation
    cd for_release
    python3 segment_libricss.py -data_path .
fi
