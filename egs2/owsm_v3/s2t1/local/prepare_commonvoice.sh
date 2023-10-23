#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=1       # start from 0 if you need to start from data preparation
stop_stage=2
SECONDS=0
utt_extra_files="text.prev text.ctc"

 . utils/parse_options.sh || exit 1;

# Note(jinchuan): language list found at: https://github.com/common-voice/common-voice/tree/main/server/data
# Some of the downloading will fail. We only process the data we can get
# Date: Jun 15, 2023
langs="ab ace af am an ar as ast az"
langs="$langs ba bas be bg bn br"
langs="$langs ca ckb cnh co cs cv cy"
langs="$langs da de dv dyu"
langs="$langs el en eo es et eu"
langs="$langs fa fi fr fy-NL"
langs="$langs ga-IE gl gn ha he hi hr hsb ht hu hy-AM"
langs="$langs ia id ig is it"
langs="$langs ja"
langs="$langs ka kab kk km kmr kn ko kpv ky"
langs="$langs lb lg lo lt lv"
langs="$langs mdf mhr mk ml mn mr mrj ms mt my myv"
langs="$langs nan-tw nb-NO ne-NP nl nn-NO nr nso"
langs="$langs oc or"
langs="$langs pa-IN pl ps pt"
langs="$langs quy"
langs="$langs rm-sursilv rm-vallader ro ru rw"
langs="$langs sah sat sc scn si sk skr sl so sq sr ss st sv-SE sw"
langs="$langs ta te tg th ti tig tk tl tn tok tr ts tt tw"
langs="$langs ug uk ur uz"
langs="$langs ve vec vi vot"
langs="$langs xh"
langs="$langs yi yo yue"
langs="$langs zgh zh-CN zh-HK zh-TW zu zza"

# base url for downloads.
# Note(jinchuan): url updated at Jun 15. 2023
data_url=https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-13.0-2023-03-09/cv-corpus-13.0-2023-03-09-${lang}.tar.gz

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p ${COMMONVOICE}
if [ -z "${COMMONVOICE}" ]; then
    log "Fill the value of 'COMMONVOICE' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for lang in langs; do
    local/download_and_untar.sh ${COMMONVOICE} ${data_url} \
      cv-corpus-13.0-2023-03-09-${lang}.tar.gz || echo "Prepare $lang fails. Continue ..."
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 local/prepare_commonvoice.py \
      --data_dir ${COMMONVOICE}/cv-corpus-13.0-2023-03-09 \
      --prefix commonvoice \
      --output_dir data/CommonVoice
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for part in train dev test; do
        utils/combine_data.sh --extra-files "text.prev text.ctc" \
            data/CommonVoice/${part} data/CommonVoice/*/${part}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
