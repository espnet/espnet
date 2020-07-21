#!/bin/bash

# Copyright 2020 Johns Hopkins University (Hao Yan )
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# data
data_url=www.openslr.org/resources/62

. ./utils/parse_options.sh || exit 1;
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${AIDATATANG}" ]; then
  log "Error: \$AIDATATANG is not set in db.sh."
  exit 2
fi

train_set=train_sp
train_dev=dev
recog_set="dev test"

echo "stage -1: Data Download"
local/download_and_untar.sh ${AIDATATANG} ${data_url} aidatatang_200zh

### Task dependent. You have to make data the following preparation part by yourself.
### But you can utilize Kaldi recipes in most cases
echo "stage 0: Data preparation"
local/data_prep.sh ${AIDATATANG}/aidatatang_200zh/corpus ${AIDATATANG}/aidatatang_200zh/transcript
# remove space in text
for x in train dev test; do
    cp data/${x}/text data/${x}/text.org
    paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
        > data/${x}/text
    rm data/${x}/text.org
done
