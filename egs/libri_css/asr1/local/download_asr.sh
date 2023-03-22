#!/usr/bin/env bash
#
# Copyright  2020-2021  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <asr-url> <asr-dir>"
  echo -e >&2 "eg:\n  $0 'https://drive.google.com/open?id=1RHYAhcnlKz08amATrf0ZOWFLzoQphtoc' download/asr_librispeech"
  exit 1
fi

asr_url=$1
asr_dir=$2

set -e -o pipefail

mkdir -p ${asr_dir}

download_from_google_drive.sh ${asr_url} ${asr_dir} "tar.gz"
