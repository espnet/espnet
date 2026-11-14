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
stage=1
stop_stage=100000
python=python3
# Pinned dataset revisions used for the published models.
fleurs_revision=70bb2e84b976b7e960aa89f1c648e09c59f894dd
cs_fleurs_revision=0cdbf166c5517ae4b6eb1c54248522eedec53017
cs_yodas_revision=e51028041b403f63c99ac91a4af040e72d0cad0e

log "$0 $*"
. utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

for corpus in FLEURS CS_FLEURS CS_YODAS; do
    if [ -z "${!corpus}" ]; then
        log "Fill the value of '${corpus}' of db.sh"
        exit 1
    fi
done
# A recipe-specific directory: egs2/fleurs/asr1 uses ${FLEURS}/<lang>
fleurs_dir=${FLEURS}/fleurs_cs_lid
cs_fleurs_dir=${CS_FLEURS}/cs-fleurs
cs_yodas_dir=${CS_YODAS}/cs-yodas

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ -s "${fleurs_dir}/test.tsv" ]; then
        log "Stage 1: ${fleurs_dir}/{train,dev,test}.tsv exist, skip the FLEURS download"
    else
        log "Stage 1: Download google/fleurs and write TSV manifests to ${fleurs_dir}"
        ${python} local/create_fleurs_tsv.py \
            --out_root "${fleurs_dir}" --cache_dir "${fleurs_dir}/cache" \
            --revision "${fleurs_revision}"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Download CS-FLEURS to ${cs_fleurs_dir}"
    local/download_cs_fleurs.sh --root "${cs_fleurs_dir}" --revision "${cs_fleurs_revision}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Download CS-YODAS to ${cs_yodas_dir}"
    local/download_cs_yodas.sh --root "${cs_yodas_dir}" --revision "${cs_yodas_revision}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Prepare data directories"
    ${python} local/prepare_fleurs_cs_lid_data.py \
        --outdir data \
        --fleurs_tsv_root "${fleurs_dir}" \
        --cs_fleurs_root "${cs_fleurs_dir}" \
        --cs_yodas_root "${cs_yodas_dir}"
    for dset in data/train_* data/valid_* data/test_*; do
        utils/validate_data_dir.sh --no-feats --non-print "${dset}" || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
