#!/usr/bin/env bash

# Copyright 2022  Shanghai Jiao Tong University (Author: Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

stage=1
stop_stage=100

sample_rate=8k

. utils/parse_options.sh
. ./db.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Enh and ASR data preparation"
    local/enh_data.sh --sample_rate ${sample_rate} --min_or_max "max"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Combine anechoic and reverberant data"
    utils/combine_data.sh --extra_files "spk1.scp spk2.scp text_spk1 text_spk2" \
        data/tr_spatialized_multi_multich_max_${sample_rate} data/tr_spatialized_anechoic_multich_max_${sample_rate} data/tr_spatialized_reverb_multich_max_${sample_rate}
    utils/combine_data.sh --extra_files "spk1.scp spk2.scp text_spk1 text_spk2" \
        data/cv_spatialized_multi_multich_max_${sample_rate} data/cv_spatialized_anechoic_multich_max_${sample_rate} data/cv_spatialized_reverb_multich_max_${sample_rate}
fi
