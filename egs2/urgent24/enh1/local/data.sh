#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>]

  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
EOF
)


stage=1
stop_stage=2

log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' in db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' in db.sh"
    exit 1
fi

if [ ! -e "${COMMONVOICE}" ]; then
    log "Please manually downlaod CommonVoice 11.0 English data from https://commonvoice.mozilla.org/en/datasets."
    log "Then set the value of 'COMMONVOICE' to the path of downloaded CommonVoice data directory in db.sh"
    exit 1
elif [ ! -e "${COMMONVOICE}/cv-corpus-11.0-2022-09-21" ]; then
    log "Please set 'COMMONVOICE' in db.sh to the path of CommonVoice data directory which contains 'cv-corpus-11.0-2022-09-21'."
    exit 1
fi


odir="${PWD}/local/"; mkdir -p "${odir}"
cwd="${PWD}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Environment preparation"

    if [ ! -d "${odir}/urgent2024_challenge" ]; then
        log "Cloning https://github.com/urgent-challenge/urgent2024_challenge to ${odir}/urgent2024_challenge"
        git clone https://github.com/urgent-challenge/urgent2024_challenge "${odir}/urgent2024_challenge"
        cd "${odir}/urgent2024_challenge"
        git submodule update --init --recursive
        cd "${cwd}"
    fi

    python -m pip install -r "${odir}/urgent2024_challenge/requirements.txt"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Downloading"

    cd "${odir}"/urgent2024_challenge

    mkdir -p wsj
    if [ ! -d "wsj/wsj0" ]; then
        ln -s "${WSJ0}" wsj/wsj0
    fi
    if [ ! -d "wsj/wsj1" ]; then
        ln -s "${WSJ1}" wsj/wsj1
    fi

    mkdir -p datasets_cv11_en
    if [ ! -d "datasets_cv11_en/cv-corpus-11.0-2022-09-21/en/clips" ]; then
        ln -s "${COMMONVOICE}/cv-corpus-11.0-2022-09-21" datasets_cv11_en/
    fi

    cd "${cwd}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"

    cd "${odir}"/urgent2024_challenge
    ./prepare_espnet_data.sh
    cd "${cwd}"
    cp -r "${odir}"/urgent2024_challenge/data "${PWD}/data"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
