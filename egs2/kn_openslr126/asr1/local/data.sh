#!/usr/bin/env bash
# Prepare IISc-MILE Kannada ASR Corpus (OpenSLR SLR126) into Kaldi-style
# data/{train,dev,test}. Stage 0 downloads + extracts; stage 1 builds dirs.
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

log "$0 $*"
. utils/parse_options.sh || exit 1
. ./path.sh || exit 1
. ./cmd.sh  || exit 1
. ./db.sh   || exit 1

if [ -z "${KANNADA:-}" ]; then
    log "Fill the value of 'KANNADA' in db.sh"
    exit 1
fi

corpus="${KANNADA}"
base_url=https://www.openslr.org/resources/126

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ -d "${corpus}/train/audio_files" ] && [ -d "${corpus}/test/audio_files" ]; then
        log "stage 0: corpus already present at ${corpus}. Skip download."
    else
        log "stage 0: Download + extract IISc-MILE Kannada ASR corpus (SLR126)"
        for split in train test; do
            if [ -d "${corpus}/${split}/audio_files" ]; then continue; fi
            mkdir -p "${corpus}/${split}"
            tarball="${corpus}/mile_kannada_${split}.tar.gz"
            if [ ! -f "${tarball}" ]; then
                wget -c -O "${tarball}" "${base_url}/mile_kannada_${split}.tar.gz"
            fi
            tar -xzf "${tarball}" -C "${corpus}/${split}"
            # flatten if the archive wrapped audio_files/ in an extra top-level dir
            if [ ! -d "${corpus}/${split}/audio_files" ]; then
                inner=$(find "${corpus}/${split}" -maxdepth 2 -type d -name audio_files | head -1)
                if [ -n "${inner}" ]; then
                    wrap=$(dirname "${inner}")
                    mv "${wrap}"/* "${corpus}/${split}/"
                    rmdir "${wrap}" 2>/dev/null || true
                fi
            fi
            if [ ! -d "${corpus}/${split}/audio_files" ] || [ ! -d "${corpus}/${split}/trans_files" ]; then
                log "Error: ${corpus}/${split} missing audio_files/trans_files after extraction"
                exit 1
            fi
        done
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Prepare data/{train,dev,test}"
    python3 local/data_prep.py -d "${corpus}" --dev-spk-percent 4 --seed 0
    for dset in train dev test; do
        utils/fix_data_dir.sh data/${dset}
        utils/validate_data_dir.sh --no-feats data/${dset}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
