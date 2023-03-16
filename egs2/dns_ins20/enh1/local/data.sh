#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] [--configure <conf.json>]
  optional argument:
    [--stage]: 1 (default) or 2
    [--stop_stage]: 1 or 2 (default)
    [--configure]: use another specific configuration file
    NOTE:
        stage 1: Create the Data Mixture from the DNS scripts. You can skip this step when you already have the audio mixture for training.
        stage 2: Prepare the data for ESPNet-se
        You can clone the DNS-interspeech2020 by git clone -b interspeech2020/master https://github.com/microsoft/DNS-Challenge.git DNS-Challenge
        If you do not want to use the default noisyspeech_synthesizer.cfg configuration under the DNS directory, you can specify your configuration file.
        Please make sure the destination is under data/dns_wav
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=2
configure=
dns_wav=$PWD/data/dns_wav


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${DNS}" ]; then
    log "Fill the value of 'DNS' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Simulation"
    if [ -z "$configure" ]; then
        local/dns_create_mixture.sh ${DNS} ${dns_wav}  || exit 1;
    else
        local/dns_create_mixture.sh --configure ${configure} ${DNS} ${dns_wav}  || exit 1;
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data preparation"
    # The following datasets will be created:
    # {tr,cv}_synthetic tt_synthetic__{no,with}_reverb
    local/dns_data_prep.sh  ${dns_wav} ${DNS}/datasets/test_set/ || exit 1;
fi
