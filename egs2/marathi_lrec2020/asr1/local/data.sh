#!/bin/bash
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=0
stop_stage=1
SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

set -e
set -u
set -o pipefail

. utils/parse_options.sh

log "Data preparation started"

if [ -z "${MARATHI_LREC2020}" ]; then
    log "Variable MARATHI_LREC2020 not set in db.sh"
    exit 2
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Preparing data for Marathi LREC2020"
    python3 local/data_prep.py -d ${MARATHI_LREC2020}
    for x in train dev test; do
        utils/spk2utt_to_utt2spk.pl data/marathi_${x}/spk2utt > data/marathi_${x}/utt2spk
        utils/fix_data_dir.sh data/marathi_${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"