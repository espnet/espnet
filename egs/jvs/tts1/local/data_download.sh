#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;

download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

if [ ! -e "${download_dir}/jvs_ver1" ]; then
    mkdir -p "${download_dir}"
    download_from_google_drive.sh \
        "https://drive.google.com/open?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt" \
        "${download_dir}"
    echo "Successfully downloaded data."
else
    echo "Already exists. Skipped."
fi
