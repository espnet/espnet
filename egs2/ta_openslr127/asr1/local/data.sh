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

stage=0
stop_stage=1
dev_ratio=0.1   # fraction of train *speakers* held out as the dev/valid set

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

. utils/parse_options.sh

if [ -z "${TAMIL}" ]; then
    log "Fill the value of 'TAMIL' in db.sh"
    exit 1
fi

mkdir -p "${TAMIL}"
corpus="${TAMIL}/mile_tamil_asr_corpus"
tarball="${TAMIL}/mile_tamil_asr_corpus.tar.gz"
data_url=https://www.openslr.org/resources/127/mile_tamil_asr_corpus.tar.gz

log "data preparation started"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -d "${corpus}/train/audio_files" ] && [ -d "${corpus}/test/audio_files" ]; then
        log "stage 0: ${corpus} already extracted. Skip download."
    else
        log "stage 0: Download + extract IISc-MILE Tamil ASR corpus to ${TAMIL}"
        if [ ! -f "${tarball}" ]; then
            wget -c -O "${tarball}" "${data_url}"
        fi
        tar -xzf "${tarball}" -C "${TAMIL}"
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Prepare data/{train_ta,dev_ta,test_ta}"
    python3 local/data_prep.py --corpus "${corpus}" --dev_ratio "${dev_ratio}"
    for x in train_ta dev_ta test_ta; do
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > data/${x}/spk2utt
        utils/fix_data_dir.sh data/${x}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
