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

DATA_PREP_ROOT=$1
TASK=${2:-"instrument"}

if [ -z "${NSYNTH}" ]; then
    log "Fill the value of 'NSYNTH' of db.sh"
    exit 1
fi


wget -P ${NSYNTH}/ http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz &
wget -P ${NSYNTH}/ http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz &
wget -P ${NSYNTH}/ http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz &

wait

for split in train valid test; do
    if [ ! -f ${NSYNTH}/nsynth-${split}.jsonwav.tar.gz ]; then
        log "File ${NSYNTH}/nsynth-${split}.jsonwav.tar.gz not found!"
        exit 1
    else
        tar -xvf "${NSYNTH}/nsynth-${split}.jsonwav.tar.gz" -C "${NSYNTH}" &
    fi
done

wait


mkdir -p ${DATA_PREP_ROOT}
python3 local/data_prep_nsynth.py --root ${NSYNTH} --task ${TASK} --output ${DATA_PREP_ROOT}

for x in train valid test; do
   for f in text wav.scp utt2spk; do
       sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
   done
   utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
   utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
done

log "Successfully finished. [elapsed=${SECONDS}s]"
