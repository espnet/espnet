. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0
lang=hi-en #hi-en oe bn-en

. utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
mkdir -p ${MUCS_subtask2}
if [ -z "${MUCS_subtask2}" ]; then
    log "Fill the value of 'MUCS_subtask2' of db.sh"
    exit 1
fi

set -e
set -u
set -o pipefail

train_set=train
train_dev=valid
test_set=valid

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage1: Download data to ${MUCS_subtask2}"
    mkdir -p ${MUCS_subtask2}
    local/download_data.sh ${MUCS_subtask2} $lang
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ ! -e ${MUCS_subtask2}/${lang}.path_done ]; then
    log "stage2: Preparing data for MUCS subtask2"
    for dset in test train; do
        local/prepare_data.sh ${MUCS_subtask2}/$lang/$dset/transcripts/wav.scp ${MUCS_subtask2}/$lang/$dset/ out.scp
      done
    touch ${MUCS_subtask2}/${lang}.path_done
    else
        echo "Path written already. Skipping."
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
