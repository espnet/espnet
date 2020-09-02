#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li, Wangyou Zhang)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh


wavdir=${PWD}/wav # set the directory of the multi-condition training WAV files to be generated

. utils/parse_options.sh


if [ ! -e "${REVERB}" ]; then
    log "Fill the value of 'REVERB' of db.sh"
    exit 1
fi
if [ ! -e "${WSJCAM0}" ]; then
    log "Fill the value of 'WSJCAM0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi


local/generate_data.sh --wavdir ${wavdir} ${WSJCAM0}
local/prepare_simu_data.sh --wavdir ${wavdir} ${REVERB} ${WSJCAM0}
local/prepare_real_data.sh --wavdir ${wavdir} ${REVERB}

# Download and install speech enhancement evaluation tools
if [ ! -d local/REVERB_scores_source ] || [ ! -d local/REVERB_scores_source/REVERB-SPEENHA.Release04Oct/evaltools/SRMRToolbox ] || [ ! -f local/PESQ ]; then
    # download and install speech enhancement evaluation tools
    local/download_se_eval_tool.sh
fi


# Additionally use WSJ clean data. Otherwise the encoder decoder is not well trained
local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
local/wsj_format_data.sh
