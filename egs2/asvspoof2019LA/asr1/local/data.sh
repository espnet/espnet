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
log "$0 $*"

. ./db.sh
. ./path.sh
. ./cmd.sh

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${ASVSPOOF2019LA}" ]; then
    log "Please fill the value of 'ASVSPOOF2019LA' of db.sh."
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Check if the data directories exists.
    if [ ! -e "${ASVSPOOF2019LA}/ASVspoof2019_LA_dev" ] && [ ! -e "${ASVSPOOF2019LA}/ASVspoof2019_LA_eval" ] && [ ! -e "${ASVSPOOF2019LA}/ASVspoof2019_LA_train" ]; then
	log "Stage 1: Download the ASVspoof2019 LA dataset and unzip it to ${ASVSPOOF2019LA}."

    # If they don't exist, download the data.
    if hash axel 2>/dev/null; then
        log "System has axel installed. Using axel..."
        axel -n 10 -o ${ASVSPOOF2019LA} https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y
    else
        log "System has no axel installed. Using wget..."
        wget -P ${ASVSPOOF2019LA} https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y
    fi

    # Unzip the dataset
    
    if hash unzip 2>/dev/null; then
        log "System has unzip installed. Using unzip..."
        unzip ${ASVSPOOF2019LA}/LA.zip -d ${ASVSPOOF2019LA}
    else
        log "System has no unzip installed. Aborting. Please unzip the dataset manually."
        echo "Dataset location: ${ASVSPOOF2019LA}/LA.zip"
    fi

    # Move the dataset to the correct location
    log "Moving the dataset to the specified location."
    mv ${ASVSPOOF2019LA}/LA/* ${ASVSPOOF2019LA}

    else # If the dataset has already been downloaded
        # Check if the protocol directory is there
        if [! -e "${ASVSPOOF2019LA}/ASVspoof2019_LA_cm_protocols"]; then
            echo "Stage 1: Missing ASVspoof2019 LA cm protocols. Unable to generate labels. Please download the ASVspoof2019 LA cm protocols and unzip it to ${ASVSPOOF2019LA}."
        else
        log "Stage 1: ${ASVSPOOF2019LA}/ASVspoof2019_LA_train, ${ASVSPOOF2019LA}/ASVspoof2019_LA_dev and ${ASVSPOOF2019LA}/ASVspoof2019_LA_eval already exists; Protocol directory present. Skipping download, do not abort."
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    log "Creating data directories..."
    mkdir -p data/{train,dev,eval}
    # Generate text, utt2spk, and wav.scp for train, dev, and eval.
    log "Generating text, utt2spk, and wav.scp for train, dev, and eval..."
    python3 local/data_prep.py ${ASVSPOOF2019LA}
    # Generate spk2utt from utt2spk.
    log "Generating spk2utt from utt2spk..."
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > "data/train/spk2utt"
    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > "data/dev/spk2utt"
    utils/utt2spk_to_spk2utt.pl data/eval/utt2spk > "data/eval/spk2utt"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"