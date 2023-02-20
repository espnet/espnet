#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
asr_exp=$1

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MSUPERB}
if [ -z "${MSUPERB}" ]; then
    log "Fill the value of 'MSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "Linguistic scoring started"
log "$0 $*"

mkdir -p downloads
if [ ! -f downloads/linguistic.json ]; then
    wget -O downloads/linguistic.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/linguistic.json?raw=true
fi
if [ ! -f downloads/macro.json ]; then
    wget -O downloads/macro.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/macro.json?raw=true
fi
if [ ! -f downloads/exception.json ]; then
    wget -O downloads/exception.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/exception.json?raw=true 
fi

python local/multilingual_analysis.py \
    --dir ${asr_exp}
