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
DATA_PREP_ROOT=${1:-"."}
shift $(( $# > 0 ? 1 : 0 ))

. ./db.sh
. ./path.sh
. ./cmd.sh

train_metadata=
valid_metadata=
test_metadata=
metadata=
valid_ratio=0.1
valid_seed=0
. utils/parse_options.sh

if [ -z "${VGGSOUND}" ]; then
    log "Fill the value of 'VGGSOUND' of db.sh"
    exit 1
fi

mkdir -p ${DATA_PREP_ROOT}
_opts=
[ -n "${metadata}" ] && _opts+=" --metadata ${metadata}"
[ -n "${train_metadata}" ] && _opts+=" --train-metadata ${train_metadata}"
[ -n "${valid_metadata}" ] && _opts+=" --valid-metadata ${valid_metadata}"
[ -n "${test_metadata}" ] && _opts+=" --test-metadata ${test_metadata}"
_opts+=" --valid-ratio ${valid_ratio}"
_opts+=" --valid-seed ${valid_seed}"
# shellcheck disable=SC2086
python3 local/data_prep_vggsound.py ${VGGSOUND} ${DATA_PREP_ROOT} ${_opts}

for x in train valid test; do
    for f in text wav.scp utt2spk; do
        sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
    utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
done

log "Successfully finished. [elapsed=${SECONDS}s]"
