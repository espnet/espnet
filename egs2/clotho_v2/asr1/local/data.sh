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

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${CLOTHO_V2}" ]; then
    log "Fill the value of 'CLOTHO_V2' of db.sh"
    exit 1
fi

## DOWNLOAD DATA
CLOTHO_V2_ROOT_DIR="$(pwd)/local"
if [ ! -e "${CLOTHO_V2_ROOT_DIR}/download_done" ]; then
    echo "Installing aac-datasets."
    if ! pip install aac-datasets; then
        echo "Error: Installing aac-datasets failed."
        exit 1
    fi
    echo "Downlaoding clotho into ${CLOTHO_V2_ROOT_DIR}."
    mkdir -p "${CLOTHO_V2_ROOT_DIR}"
    for split in val eval; do
        echo "Downloading ${split} split."
        if ! aac-datasets-download --root "${CLOTHO_V2_ROOT_DIR}" clotho --subsets "${split}"; then
            echo "Error: Downloading Clotho dataset failed."
            exit 1
        fi
    done
    touch "${CLOTHO_V2_ROOT_DIR}/download_done"
else
    echo "Clotho dataset is already downloaded. ${CLOTHO_V2_ROOT_DIR}/download_done exists."
fi

SPLITS=(development validation evaluation) # map dev, val, eval to development, validation, evaluation
N_REF=5
CLOTHO_V2_ROOT_DIR="${CLOTHO_V2_ROOT_DIR}/CLOTHO_v2.1"


## PREPARE DATA
log "stage 1: Data preparation"
for split_name in ${SPLITS[@]}; do
    mkdir -p "data/${split_name}"
done

if [ ! -d ${CLOTHO_V2_ROOT_DIR} ]; then
    echo Cannot find CLOTHO_V2_ROOT_DIR root! Exiting...
    exit 1
fi

# Prepare data in the Kaldi format, including three files:
# text, wav.scp, utt2spk
echo "$(which python)"
python3 local/data_prep.py ${CLOTHO_V2_ROOT_DIR} ${N_REF}

for split_name in ${SPLITS[@]}; do
    for f in wav.scp utt2spk; do
        sort data/${split_name}/${f} -o data/${split_name}/${f}
    done
    # Sort all text files
    if [ -f data/${split_name}/text ]; then
        sort data/${split_name}/text -o data/${split_name}/text
    fi
    for i in $(seq ${N_REF}); do
        if [ -f data/${split_name}/text_spk${i} ]; then
            sort data/${split_name}/text_spk${i} -o data/${split_name}/text_spk${i}
        fi
    done
    echo "Running spk2utt for ${split_name}"
    utils/utt2spk_to_spk2utt.pl data/${split_name}/utt2spk > "data/${split_name}/spk2utt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"