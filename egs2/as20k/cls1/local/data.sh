#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

DATA_PREP_ROOT=${1:-"."}

if [ -z "${AUDIOSET}" ]; then
    log "Fill the value of 'AUDIOSET' of db.sh"
    exit 1
fi

mkdir -p ${DATA_PREP_ROOT}
python3 local/data_prep_as20k.py ${AUDIOSET} ${DATA_PREP_ROOT}

for x in val eval train; do
    for f in text wav.scp utt2spk; do
        sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
    utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
done

log "Successfully finished. [elapsed=${SECONDS}s]"
