set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100

data_url=https://web.eecs.umich.edu/~mihalcea/downloads/
data_url2=https://huggingface.co/datasets/declare-lab/MELD/resolve/main/


log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 1 ]; then
    log "Usage: $0 <datadir>"
    exit 2
fi

DATA_PREP_ROOT=$1

if [ -z "${MELD}" ]; then
    log "Fill the value of 'MELD' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Download"
    mkdir -p ${MELD}
    if ! local/download_and_untar.sh --remove-archive ${MELD} ${data_url}; then
        log "Failed to download from the original site, try a backup site."
        local/download_and_untar.sh --remove-archive ${MELD} ${data_url2}
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p ${DATA_PREP_ROOT}/{train,valid,test}
    python3 local/data_prep.py ${MELD} ${DATA_PREP_ROOT}
    for x in test valid train; do
        for f in text wav.scp utt2spk; do
            sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
