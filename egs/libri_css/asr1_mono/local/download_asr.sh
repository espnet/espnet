#!/usr/bin/env bash
#
# Copyright  2020  University of Stuttgart (Author: Pavel Denisov)
# Apache 2.0

# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 0 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0"
  exit 1
fi


set -e -o pipefail

mkdir -p downloads
dir=$(mktemp -d -p downloads)

download_from_google_drive.sh \
	"https://drive.google.com/open?id=17cOOSHHMKI82e1MXj4r2ig8gpGCRmG2p" \
	${dir}  ".tar.gz"

rm -f ${dir}/*.tar.gz
cp -a ${dir}/* .
rm -rf ${dir}

