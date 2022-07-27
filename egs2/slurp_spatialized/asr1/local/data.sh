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
stop_stage=3
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${SLURP}" ]; then
    log "Fill the value of 'SLURP' of db.sh"
    exit 1
fi

if [ -z "${SLURP_S}" ]; then
    log "Fill the value of 'SLURP_S' of db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${SLURP}/LICENSE.txt" ]; then
    	log "Data Preparation stage 1: Download data to ${SLURP}"
        git clone https://github.com/pswietojanski/slurp.git ${SLURP}
    else
        log "Data Preparation stage 1: ${SLURP}/LICENSE.txt is already existing. Skip data downloading"
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ ! -d data/local/SpatializedSLURP_SC ]; then
    	log "Data Preparation stage 2: Create Single-Channel SLURP-S Mixture"
         python3 local/multi_to_single.py ${SLURP_S} data/local/SpatializedSLURP_SC
    else
        log "Data Preparation stage 2: data/local/SpatializedSLURP_SC is already existing. Skip Creating Single-Channel SLURP-S Mixture"
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Data Preparation stage 3: Data Preparation"
    mkdir -p data/{train,valid,test}
    python3 local/prepare_slurp_data.py ${SLURP}
    cp -r data/test data/test_qut
    
    for x in test test_qut devel train; do
        # Write wav.scp base on raw_wav.scp
        log "Data Preparation stage 3: Data Preparation ${x}"
        python3 local/prepare_slurp_mixture.py data/${x} data/local/SpatializedSLURP_SC  ${x}
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
    local/run_spm.sh
    mv data data_old
    mv data_bpe_500 data
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
