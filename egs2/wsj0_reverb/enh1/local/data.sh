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



mkdir -p ./data/

local/convert2wav.sh ${WSJ0} ./data/wsj0_ful_wav || exit 1;

pip3 install -q pyroomacoustics==0.6.0

python local/create_wsj0_reverb.py --wsj0_dir ./data/wsj0_ful_wav/wsj0 --target_dir ./data/wsj_reverb

WSJ_REVERB=./data/wsj_reverb/audio

for f in test valid train;
do

mkdir -p data/${f}
find ${WSJ_REVERB}/${f}/reverb/ -iname '*.wav' | sort > data/${f}/wav.id.scp
find ${WSJ_REVERB}/${f}/anechoic/ -iname '*.wav' | sort > data/${f}/spk1.id.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp) > data/${f}/wav.scp
paste <(cat data/${f}/spk1.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/spk1.id.scp) > data/${f}/spk1.scp

paste <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) <(cat data/${f}/wav.id.scp | xargs -L 1  basename -s .wav ) > data/${f}/utt2spk
cp data/${f}/utt2spk data/${f}/spk2utt

done
