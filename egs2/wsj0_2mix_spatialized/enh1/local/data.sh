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

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. ./db.sh

local/wsj0_2mix_data.sh --min_or_max ${min_or_max} --sample_rate ${sample_rate}

# wsj_full_wav=$PWD/data/wsj0/wsj0_wav
wsj_2mix_wav=$PWD/data/wsj0_mix
wsj_2mix_spatialized_wav=$PWD/data/wsj0_mix_spatialized
wsj_2mix_spatialized_scripts=$PWD/data/wsj0_mix_spatialized/scripts

# train_set: tr_${min_or_max}_${sample_rate}_spatialized_anechoic_multich
#            tr_${min_or_max}_${sample_rate}_spatialized_reverb_multich
# train_dev: cv_${min_or_max}_${sample_rate}_spatialized_anechoic_multich
#            cv_${min_or_max}_${sample_rate}_spatialized_reverb_multich
# recog_set: tt_${min_or_max}_${sample_rate}_spatialized_anechoic_multich
#            tt_${min_or_max}_${sample_rate}_spatialized_reverb_multich

### This part is for spatializing WSJ0 mix
### Download spatialize_mixture scripts and spatialize mixtures for 2 speakers
local/spatialize_wsj0_mix.sh --min_or_max ${min_or_max} --sample_rate ${sample_rate} \
    ${wsj_2mix_spatialized_scripts} ${wsj_2mix_wav} ${wsj_2mix_spatialized_wav} || exit 1;
local/wsj0_2mix_spatialized_data_prep.sh --min_or_max ${min_or_max} \
    --sample_rate ${sample_rate} data ${wsj_2mix_spatialized_wav} || exit 1;

### create .scp file for reference audio
for x in tr_spatialized_anechoic_multich cv_spatialized_anechoic_multich tt_spatialized_anechoic_multich \
         tr_spatialized_reverb_multich cv_spatialized_reverb_multich tt_spatialized_reverb_multich; do
    sed -e 's/\/mix\//\/s1\//g' ./data/$x/wav.scp > ./data/$x/spk1.scp
    sed -e 's/\/mix\//\/s2\//g' ./data/$x/wav.scp > ./data/$x/spk2.scp
done
