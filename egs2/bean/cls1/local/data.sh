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

DATA_PREP_ROOT=${1:-"."}
DATASETS=${2:-"watkins bats cbi humbugdb dogs"}

if [ -z "${BEAN}" ]; then
    log "Fill the value of 'BEAN' in db.sh"
    exit 1
fi

# Watkins-Whales
if [[ "${DATASETS}" == *"watkins"* ]]; then
    log "Processing watkins"
    WATKINS_LOCATION="${DATA_PREP_ROOT}/downloads/watkins"
    if [ "${BEAN}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/watkins/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/watkins.zip -P ${DATA_PREP_ROOT}/downloads
            unzip ${DATA_PREP_ROOT}/downloads/watkins.zip -d ${WATKINS_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/watkins.zip
            touch "${WATKINS_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEAN}/watkins"
        WATKINS_LOCATION="${BEAN}/watkins"
    fi
    python local/scripts/watkins.py ${WATKINS_LOCATION} ${DATA_PREP_ROOT}
fi


# Bats
if [[ "${DATASETS}" == *"bats"* ]]; then
    log "Processing bats"
    BATS_LOCATION="${DATA_PREP_ROOT}/downloads/bats"
    if [ "${BEAN}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/bats/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip -P ${DATA_PREP_ROOT}/downloads
            unzip ${DATA_PREP_ROOT}/downloads/egyptian_fruit_bats.zip -d ${BATS_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/egyptian_fruit_bats.zip
            touch "${BATS_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEAN}/bats"
        BATS_LOCATION="${BEAN}/bats"
    fi
    python local/scripts/bats.py ${BATS_LOCATION} ${DATA_PREP_ROOT}
fi


# CBI-Birds
if [[ "${DATASETS}" == *"cbi"* ]]; then
    log "Processing cbi"
    CBI_LOCATION="${DATA_PREP_ROOT}/downloads/cbi"
    if [ "${BEAN}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/cbi/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            pip install kaggle==1.5.12
            kaggle competitions download -p ${DATA_PREP_ROOT}/downloads birdsong-recognition
            unzip ${DATA_PREP_ROOT}/downloads/birdsong-recognition.zip -d ${CBI_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/birdsong-recognition.zip
            touch "${CBI_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEAN}/cbi"
        CBI_LOCATION="${BEAN}/cbi"
    fi
    python local/scripts/cbi.py ${CBI_LOCATION} ${DATA_PREP_ROOT}
fi


# Humbugdb-Mosquito
if [[ "${DATASETS}" == *"humbugdb"* ]]; then
    log "Processing humbugdb"
    HUMBUGDB_LOCATION="${DATA_PREP_ROOT}/downloads/humbugdb"
    if [ "${BEAN}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/humbugdb/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            git clone https://github.com/HumBug-Mosquito/HumBugDB.git ${HUMBUGDB_LOCATION}
            for i in 1 2 3 4; do
                wget -O ${DATA_PREP_ROOT}/downloads/humbugdb_neurips_2021_${i}.zip https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_${i}.zip?download=1
                unzip ${DATA_PREP_ROOT}/downloads/humbugdb_neurips_2021_${i}.zip -d ${HUMBUGDB_LOCATION}/data/audio
                rm ${DATA_PREP_ROOT}/downloads/humbugdb_neurips_2021_${i}.zip
            done
            touch "${HUMBUGDB_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEAN}/humbugdb"
        HUMBUGDB_LOCATION="${BEAN}/humbugdb"
    fi
    python local/scripts/humbugdb.py ${HUMBUGDB_LOCATION} ${DATA_PREP_ROOT}
fi


# Dogs
if [[ "${DATASETS}" == *"dogs"* ]]; then
    log "Processing dogs"
    DOGS_LOCATION="${DATA_PREP_ROOT}/downloads/dogs"
    if [ "${BEAN}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/dogs/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip -P ${DATA_PREP_ROOT}/downloads
            unzip ${DATA_PREP_ROOT}/downloads/dog_barks.zip -d ${DOGS_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/dog_barks.zip
            # Change to mono channel
            audio_dir="${DOGS_LOCATION}/audio"
            mkdir -p "${audio_dir}.tmp"
            mv "${audio_dir}"/* "${audio_dir}.tmp/"
            for file in "$audio_dir".tmp/*; do
                filename=$(basename "$file")
                set +e # don't care about few corrupt files
                sox "$file" -c 1 "${audio_dir}/${filename}"
                set -e
            done
            rm -r "${audio_dir}.tmp"
            
            touch "${DOGS_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEAN}/dogs"
        DOGS_LOCATION="${BEAN}/dogs"
    fi
    python local/scripts/dogs.py ${DOGS_LOCATION} ${DATA_PREP_ROOT}
fi


for dataset in ${DATASETS}; do
    for x in ${dataset}.dev ${dataset}.train ${dataset}.test; do
        for f in text wav.scp utt2spk; do
            if [ -f "${DATA_PREP_ROOT}/${x}/${f}" ]; then
                sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
            fi
        done
        if [ -f "${DATA_PREP_ROOT}/${x}/utt2spk" ]; then
            utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
            utils/validate_data_dir.sh --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
        fi
    done
done


log "Successfully finished. [elapsed=${SECONDS}s]"