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
    [--configure]: use specific configuration file 
    NOTE:
        You can clone the DNS-interspeech2020 by git clone -b interspeech2020/master https://github.com/microsoft/DNS-Challenge.git DNS-Challenge
        Default configuration is noisyspeech_synthesizer.cfg under the DNS-Challenge directory
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=1
stop_stage=2
configure=
dns_wav=$PWD/data/dns_wav
dns_script=$PWD/data/dns


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

    # Note:
    #   - `both`: a mixture of speech1, speech2 and noise (for speech separation)
    #   - `clean`: a mixture of speech1 and speech2 (for speech separation)
    #   - `single`: a mixture of speech1 and noise (for speech enhancement)
fi

