#!/usr/bin/env bash
# HEROICO (LDC2006S37) download + data preparation.
# Stage 0 downloads + extracts the corpus; stage 1 builds Kaldi-style
# data/{train,dev,test}.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]:-main}) $*"
}
SECONDS=0

stage=0
stop_stage=1
# OpenSLR mirror of the HEROICO corpus (SLR39).
data_url=https://openslr.trmal.net/resources/39/LDC2006S37.tar.gz

log "$0 $*"
. utils/parse_options.sh || exit 1
. ./path.sh || exit 1
. ./cmd.sh  || exit 1
. ./db.sh   || exit 1

if [ -z "${HEROICO:-}" ]; then
    log "Fill the value of 'HEROICO' in db.sh"
    exit 1
fi

# ${HEROICO} is the directory that contains (or will receive) the extracted
# LDC2006S37/ corpus root (the dir with data/speech and data/transcripts).
corpus="${HEROICO}/LDC2006S37"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -d "${corpus}/data/speech" ] && [ -d "${corpus}/data/transcripts" ]; then
        log "stage 0: corpus already present at ${corpus}. Skip download."
    else
        log "stage 0: Download + extract HEROICO (LDC2006S37, OpenSLR SLR39)"
        mkdir -p "${HEROICO}"
        tarball="${HEROICO}/LDC2006S37.tar.gz"
        if [ ! -f "${tarball}" ]; then
            if command -v wget >/dev/null 2>&1; then
                wget -c -O "${tarball}.tmp" "${data_url}"
            elif command -v curl >/dev/null 2>&1; then
                curl -L -C - -o "${tarball}.tmp" "${data_url}"
            else
                log "Error: neither wget nor curl is available to download the corpus."
                exit 1
            fi
            mv "${tarball}.tmp" "${tarball}"
        fi
        tar -xzf "${tarball}" -C "${HEROICO}"
        if [ ! -d "${corpus}/data/speech" ] || [ ! -d "${corpus}/data/transcripts" ]; then
            log "Error: ${corpus} is missing data/speech or data/transcripts after extraction"
            exit 1
        fi
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Prepare data/{train,dev,test}"
    if ! command -v sox >/dev/null 2>&1; then
        log "Warning: sox is not installed. wav.scp uses sox to resample 22050 Hz -> 16000 Hz,"
        log "         so sox must be available before feature extraction / training."
    fi
    python3 local/prepare_data.py --data_dir "${corpus}" --output_dir data
    log "Utterance counts per split:"
    for split in train dev test; do
        if [ -f "data/${split}/wav.scp" ]; then
            n=$(wc -l < "data/${split}/wav.scp")
            log "  ${split}: ${n} utterances"
        fi
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
