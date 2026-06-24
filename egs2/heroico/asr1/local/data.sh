#!/usr/bin/env bash
# HEROICO (LDC2006S37) data preparation wrapper.
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-LDC2006S37>"
    exit 2
fi

heroico_root="$1"

if [ ! -d "${heroico_root}" ]; then
    log "Error: dataset path does not exist: ${heroico_root}"
    exit 1
fi

if ! command -v sox >/dev/null 2>&1; then
    log "Warning: sox is not installed. wav.scp uses sox to resample 22050 Hz -> 16000 Hz,"
    log "         so sox must be available before feature extraction / training."
fi

log "Preparing HEROICO data from ${heroico_root}"
python3 local/prepare_data.py --data_dir "${heroico_root}" --output_dir data

log "Utterance counts per split:"
for split in train dev test; do
    if [ -f "data/${split}/wav.scp" ]; then
        n=$(wc -l < "data/${split}/wav.scp")
        log "  ${split}: ${n} utterances"
    fi
done

log "Successfully finished. [elapsed=${SECONDS}s]"
