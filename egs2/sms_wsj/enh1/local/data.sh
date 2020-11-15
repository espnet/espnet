#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Note: This script is based on the original repository of SMS-WSJ:
#  https://github.com/fgnt/sms_wsj
# and the sms_wsj recipe in Asteroid:
#  https://github.com/mpariente/asteroid/blob/master/egs/sms_wsj/CaCGMM/local/prepare_data.sh
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

. ./db.sh


nj=16
min_or_max=min
sample_rate=8k
download_rir=true

. utils/parse_options.sh

. ./path.sh


wsj_zeromean_wav=$PWD/data/sms_wsj/wsj_${sample_rate}_zeromean
sms_wsj_wav=$PWD/data/sms_wsj/2speakers
sms_wsj_scripts=$PWD/local/sms_wsj
other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt

if [ ! -e "${WSJ0}" ]; then
    log "Fill the value of 'WSJ0' of db.sh"
    exit 1
fi
if [ ! -e "${WSJ1}" ]; then
    log "Fill the value of 'WSJ1' of db.sh"
    exit 1
fi


if [[ ! -d ${sms_wsj_scripts} ]]; then
    log "Cloning and installing SMS-WSJ repository"
    git clone https://github.com/fgnt/sms_wsj.git ${sms_wsj_scripts}
    python -m pip install -e ${sms_wsj_scripts}
    if ${download_rir}; then
        git clone https://github.com/boeddeker/rir-generator.git ${sms_wsj_scripts}/reverb/rirgen_rep
	    python -m pip install -e ${sms_wsj_scripts}/reverb/rirgen_rep/python/
    fi
    # for reproducing the exact simulation
    if ! python -c 'import sacred' &> /dev/null; then
        log "Installing 'sacred' (required for generating SMS-WSJ)"
        python -m pip install sacred
    fi
    # Note: MPI pre-installation is required here.
    if ! python -c 'import dlp_mpi' &> /dev/null; then
        log "Installing 'dlp_mpi' (required for generating SMS-WSJ)"
        python -m pip install dlp_mpi
    fi
fi

local/create_database.sh \
    --nj ${nj} \
    --min-or-max ${min_or_max} \
    --sample-rate ${sample_rate} \
    --download-rir ${download_rir} \
    ${WSJ0} ${WSJ1} ${wsj_zeromean_wav} ${sms_wsj_wav} || exit 1;

# The following datasets will be created:
# {tr,cv,tt}_mix_{both,clean,single}_{anechoic,reverb}_${min_or_max}_${sample_rate}
#
# Note:
#   - `both`: a mixture of speech1, speech2 and noise (for speech separation)
#   - `clean`: a mixture of speech1 and speech2 (for speech separation)
#   - `single`: a mixture of speech1 and noise (for speech enhancement)
local/sms_wsj_data_prep.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
    ${sms_wsj_scripts}/whamr_scripts ${sms_wsj_wav} ${wsj_zeromean_wav} || exit 1;


### Also need wsj corpus to prepare language information
### This is from Kaldi WSJ recipe
log "local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?"
local/wsj_data_prep.sh ${WSJ0}/??-{?,??}.? ${WSJ1}/??-{?,??}.?
log "local/wsj_format_data.sh"
local/wsj_format_data.sh
log "mkdir -p data/wsj"
mkdir -p data/wsj
log "mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj"
mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/wsj


log "Prepare text from lng_modl dir: ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z -> ${other_text}"
mkdir -p "$(dirname ${other_text})"

# NOTE(kamo): Give utterance id to each texts.
zcat ${WSJ1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z | \
    grep -v "<" | tr "[:lower:]" "[:upper:]" | \
    awk '{ printf("wsj1_lng_%07d %s\n",NR,$0) } ' > ${other_text}


log "Create non linguistic symbols: ${nlsyms}"
cut -f 2- data/wsj/train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
cat ${nlsyms}
