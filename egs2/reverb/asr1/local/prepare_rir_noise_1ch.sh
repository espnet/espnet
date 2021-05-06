#!/usr/bin/env bash
set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
help_message=$(cat << EOF
Usage: $0
EOF
)


log "$0 $*"
. ./utils/parse_options.sh
. ./path.sh


if [ $# -ne 0 ]; then
  log "${help_message}"
  exit 2
fi

downloaddir=data/local
if [ ! -e "${downloaddir}"/reverb_tools/reverb_tools_for_Generate_mcTrainData ]; then
    if [ ! -e "${downloaddir}"/downloads/REVERB_TOOLS_FOR_ASR_ver2.0.tgz ]; then
        wget http://reverb2014.dereverberation.com/tools/REVERB_TOOLS_FOR_ASR_ver2.0.tgz -P "${downloaddir}"/downloads
    fi
    tar xf "${downloaddir}"/downloads/REVERB_TOOLS_FOR_ASR_ver2.0.tgz -C "${downloaddir}"
fi


for nr in noise rir; do
    o="data/reverb_${nr}_single"
    mkdir -p "${o}/data"
    for w in "${downloaddir}/reverb_tools/reverb_tools_for_Generate_mcTrainData/${nr^^}"/*.wav; do
        for ch in 1 2 3 4 5 6 7 8; do
            sox "${w}" "${o}/data/$(basename ${w%*.wav})_${ch}ch.wav" remix "${ch}"
            echo "$(basename ${w%*.wav})_${ch}ch" "${o}/data/$(basename ${w%*.wav})_${ch}ch.wav" 1>&3
        done
    done 3> "${o}/wav.scp"
done

log "Successfully finished. [elapsed=${SECONDS}s]"
