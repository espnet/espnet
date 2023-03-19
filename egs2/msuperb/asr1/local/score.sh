#!/usr/bin/env bash

# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
lid=$1
only_lid=$2
asr_exp=$3

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${MSUPERB}
if [ -z "${MSUPERB}" ]; then
    log "Fill the value of 'MSUPERB' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


log "Linguistic scoring started"
log "$0 $*"

# . utils/parse_options.sh || exit 1;

mkdir -p downloads
if [ ! -f downloads/linguistic.json ]; then
    wget -O downloads/linguistic.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/linguistic.json?raw=true
fi
if [ ! -f downloads/macro.json ]; then
    wget -O downloads/macro.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/macro.json?raw=true
fi
if [ ! -f downloads/exception.json ]; then
    wget -O downloads/exception.json https://github.com/hhhaaahhhaa/LinguisticTree/blob/main/exception.json?raw=true
fi

python local/split_results.py \
    --dir ${asr_exp} \
    --lid ${lid} \
    --only_lid ${only_lid}

if "${only_lid}" || "${lid}"; then
    # directories=$(find ${asr_exp} -wholename "*/*/score_lid/independent/*" -type d -not -path '/\.')
    # directories+=" "
    directories=$(find ${asr_exp} -wholename "*/*/score_lid/few_shot/*" -type d -not -path '/\.')
    # directories+=" "
    # directories+=$(find ${asr_exp} -wholename "*/*/score_lid/language_family/*" -type d -not -path '/\.')
    # directories+=" "
    # directories+=$(find ${asr_exp} -wholename "*/*/score_lid/all/*" -type d -not -path '/\.')
    for _scoredir in ${directories}
    do
        log "Write result in ${_scoredir}/scores.txt"
        python local/lid.py --dir ${_scoredir}
        cat "${_scoredir}/scores.txt"
    done
fi

if ! "${only_lid}"; then
    # directories=$(find ${asr_exp} -wholename "*/*/score_cer/independent/*" -type d -not -path '/\.')
    # directories+=" "
    directories=$(find ${asr_exp} -wholename "*/*/score_cer/few_shot/*" -type d -not -path '/\.')
    # directories+=" "
    # directories+=$(find ${asr_exp} -wholename "*/*/score_cer/language_family/*" -type d -not -path '/\.')
    # directories+=" "
    # directories+=$(find ${asr_exp} -wholename "*/*/score_cer/all/*" -type d -not -path '/\.')
    for _scoredir in ${directories}
    do
        log "Write result in ${_scoredir}/result.txt"
        sclite \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"
        grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
    done
fi
