#!/usr/bin/env bash

# Copyright 2021 Takenori Yoshimura
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Download JMD Corpus

. ./path.sh || exit 1

download_dir=$1
dialect=$(echo "$2" | tr '[:upper:]' '[:lower:]')

# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <download_dir> <dialect>"
    exit 1
fi

set -euo pipefail

case "${dialect}" in
    "kumamoto") url="https://drive.google.com/a/g.sp.m.is.nagoya-u.ac.jp/uc?id=1gacw6Ak6rlEZ_gx9KwafIIfc3dU0EAHW" ;;
    "osaka") url="https://drive.google.com/a/g.sp.m.is.nagoya-u.ac.jp/uc?id=1mCbmUKVifEEEcm7A3ofqWW7dCqVXGrsh" ;;
    *) echo "Given dialect is not supported" ; exit 1 ;;
esac

if [ ! -e "${download_dir}/${dialect}" ]; then
    scripts/utils/download_from_google_drive.sh \
        "${url}" "${download_dir}" zip
    echo "Successfully downloaded JMD corpus."
else
    echo "Already exists. Skipped."
fi

cwd=$(pwd)
if [ ! -e "${download_dir}/JMDComplements" ]; then
    echo "Downloading complements for ${dialect}"
    cd "${download_dir}"
    git clone https://github.com/takenori-y/JMDComplements
    cd "${cwd}"
    echo "Successfully downloaded JMD complements."
else
    echo "Already exists. Skipped."
fi
cp "${download_dir}/JMDComplements/${dialect}/segments" "${download_dir}/${dialect}/"
