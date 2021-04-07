#!/usr/bin/env bash

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

set -euo pipefail

# download dataset
if [ ! -e "${db}/CSMSC" ]; then
    echo "Now CSMSC is not free, you cannot download it anymore."
    echo "You need to apply the form: https://www.data-baker.com/open_source.html"
    echo "After you get the corpus, please locate it as follows and then re-run the recipe:"
    cat << EOF
${db}/CSMSC
├── PhoneLabeling
├── ProsodyLabeling
└── Wave
EOF
    exit 1;
else
    echo "Already exists. Skip download."
fi
