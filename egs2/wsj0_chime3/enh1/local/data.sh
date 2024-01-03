#!/bin/bash

. ./path.sh
. ./db.sh

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0
(No options)
EOF
)

if [ $# -ne 0 ]; then
    log "Error: invalid command line arguments"
    log "${help_message}"
    exit 1
fi


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi

if [ ! -e "${CHIME3}" ]; then
    log "Fill the value of 'CHIME3' of db.sh"
    exit 1
fi

mkdir -p ./data/

local/convert2wav.sh ${WSJ0} ./data/wsj0_ful_wav || exit 1;

python local/create_wsj0_chime3.py ./data/wsj0_ful_wav/wsj0 ${CHIME3}/data/ ./data/wsj_chime3

WSJ_CHIME3="./data/wsj_chime3"

for f in test valid train;
do

mkdir -p data/${f}
find ${WSJ_CHIME3}/${f}/noisy/ -iname '*.wav' | sort > data/${f}/wav.id.scp
find ${WSJ_CHIME3}/${f}/clean/ -iname '*.wav' | sort > data/${f}/spk1.id.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp) > data/${f}/wav.scp
paste <(cat data/${f}/spk1.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/spk1.id.scp) > data/${f}/spk1.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) > data/${f}/utt2spk
cp data/${f}/utt2spk data/${f}/spk2utt

done
