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
DATASETS=${2:-"dcase gibbons enabirds rfcx hiceas watkins bats cbi humbugdb dogs"}

if [ -z "${BEANS}" ]; then
    log "Fill the value of 'BEANS' in db.sh"
    exit 1
fi

### SOUND EVENT DETECTION ####
# DCASE
if [[ "${DATASETS}" == *"dcase"* ]]; then
    log "Processing DCASE 21"
    DCASE_LOCATION="${DATA_PREP_ROOT}/downloads/dcase"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${DATA_PREP_ROOT}/downloads/dcase/download.done" ]; then
            log "Skip downloading because download.done exists"
        else
            log "Downloading"
            mkdir -p ${DATA_PREP_ROOT}/{wav,downloads}
            wget -O ${DATA_PREP_ROOT}/downloads/Development_Set.zip https://zenodo.org/record/5412896/files/Development_Set.zip?download=1
            unzip ${DATA_PREP_ROOT}/downloads/Development_Set.zip -d ${DCASE_LOCATION}
            rm ${DATA_PREP_ROOT}/downloads/Development_Set.zip
            touch "${DCASE_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/dcase"
        DCASE_LOCATION="${BEANS}/dcase"
    fi
    python local/scripts/dcase.py ${DCASE_LOCATION} ${DATA_PREP_ROOT}
fi

# GIBBONS
if [[ "${DATASETS}" == *"gibbons"* ]]; then
    log "Processing gibbons"
    GIBBONS_LOCATION="${DATA_PREP_ROOT}/downloads/gibbons/original"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${GIBBONS_LOCATION}/download.done" ]; then
            log "Skip downloading gibbons data because download.done exists"
        else
            log "Downloading gibbons"
            mkdir -p ${GIBBONS_LOCATION}
            wget https://zenodo.org/record/3991714/files/Train.zip?download=1 -O ${GIBBONS_LOCATION}/hainan_gibbons_wav.zip
            unzip ${GIBBONS_LOCATION}/hainan_gibbons_wav.zip -d ${GIBBONS_LOCATION}
            rm ${GIBBONS_LOCATION}/hainan_gibbons_wav.zip
            # Download also labels
            wget https://zenodo.org/record/3991714/files/Train_Labels.zip?download=1 -O ${GIBBONS_LOCATION}/hainan_gibbons_labels.zip
            unzip ${GIBBONS_LOCATION}/hainan_gibbons_labels.zip -d ${GIBBONS_LOCATION}
            rm ${GIBBONS_LOCATION}/hainan_gibbons_labels.zip


            touch "${GIBBONS_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/gibbons"
        GIBBONS_LOCATION="${BEANS}/gibbons"
    fi
    python local/scripts/gibbons.py ${GIBBONS_LOCATION} ${DATA_PREP_ROOT}
fi

#ENABIRDS
if [[ "${DATASETS}" == *"enabirds"* ]]; then
    log "Processing enabirds"
    ENA_LOCATION="${DATA_PREP_ROOT}/downloads/enabirds/original"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${ENA_LOCATION}/download.done" ]; then
            log "Skip downloading enabirds data because download.done exists"
        else
            log "Downloading enabirds"
            mkdir -p ${ENA_LOCATION}
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/enabirds_wav.zip -O ${ENA_LOCATION}/wav_Files.zip
            unzip ${ENA_LOCATION}/wav_Files.zip -d ${ENA_LOCATION}
            rm ${ENA_LOCATION}/wav_Files.zip
            # Download also labels
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/enabirds_annotation.zip -O ${ENA_LOCATION}/enabirds_annotation.zip
            unzip ${ENA_LOCATION}/enabirds_annotation.zip -d ${ENA_LOCATION}
            rm ${ENA_LOCATION}/enabirds_annotation.zip
            touch "${ENA_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/enabirds"
        ENA_LOCATION="${BEANS}/enabirds"
    fi
    python local/scripts/enabirds.py ${ENA_LOCATION} ${DATA_PREP_ROOT}
fi

#RFCX
if [[ "${DATASETS}" == *"rfcx"* ]]; then
    log "Processing rfcx"
    RFCX_LOCATION="${DATA_PREP_ROOT}/downloads/rfcx/original"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${RFCX_LOCATION}/download.done" ]; then
            log "Skip downloading rfcx data because download.done exists"
        else
            log "Downloading rcfx"
            mkdir -p ${RFCX_LOCATION}
            kaggle competitions download -c rfcx-species-audio-detection -p $RFCX_LOCATION
            unzip ${RFCX_LOCATION}/rfcx-species-audio-detection.zip -d ${RFCX_LOCATION}
            rm ${RFCX_LOCATION}/rfcx-species-audio-detection.zip
            touch "${RFCX_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/rcfx"
        RFCX_LOCATION="${BEANS}/rcfx"
    fi
    python local/scripts/rcfx.py ${RFCX_LOCATION} ${DATA_PREP_ROOT}
fi


#HICEAS
if [[ "${DATASETS}" == *"hiceas"* ]]; then
    log "Processing hiceas"
    HI_LOCATION="${DATA_PREP_ROOT}/downloads/hiceas/original"
    if [ "${BEANS}" == "downloads" ]; then
        if [ -f "${HI_LOCATION}/download.done" ]; then
            log "Skip downloading hiceas data because download.done exists"
        else
            log "Downloading hiceas"
            mkdir -p ${HI_LOCATION}
            wget https://storage.googleapis.com/ml-bioacoustics-datasets/hiceas_1-20_minke-detection.zip -O ${HI_LOCATION}/hiceas.zip
            unzip ${HI_LOCATION}/hiceas.zip -d ${HI_LOCATION}
            rm ${HI_LOCATION}/hiceas.zip
            touch "${HI_LOCATION}/download.done"
        fi
    else
        log "Using data from the provided location: ${BEANS}/hiceas"
        HI_LOCATION="${BEANS}/hiceas"
    fi
    python local/scripts/hiceas.py ${HI_LOCATION} ${DATA_PREP_ROOT}
fi

### SOUND EVENT CLASSIFICATION ####

# Watkins-Whales
if [[ "${DATASETS}" == *"watkins"* ]]; then
    log "Processing watkins"
    WATKINS_LOCATION="${DATA_PREP_ROOT}/downloads/watkins"
    if [ "${BEANS}" == "downloads" ]; then
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
        log "Using data from the provided location: ${BEANS}/watkins"
        WATKINS_LOCATION="${BEANS}/watkins"
    fi
    python local/scripts/watkins.py ${WATKINS_LOCATION} ${DATA_PREP_ROOT}
fi


# Bats
if [[ "${DATASETS}" == *"bats"* ]]; then
    log "Processing bats"
    BATS_LOCATION="${DATA_PREP_ROOT}/downloads/bats"
    if [ "${BEANS}" == "downloads" ]; then
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
        log "Using data from the provided location: ${BEANS}/bats"
        BATS_LOCATION="${BEANS}/bats"
    fi
    python local/scripts/bats.py ${BATS_LOCATION} ${DATA_PREP_ROOT}
fi


# CBI-Birds
if [[ "${DATASETS}" == *"cbi"* ]]; then
    log "Processing cbi"
    CBI_LOCATION="${DATA_PREP_ROOT}/downloads/cbi"
    if [ "${BEANS}" == "downloads" ]; then
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
        log "Using data from the provided location: ${BEANS}/cbi"
        CBI_LOCATION="${BEANS}/cbi"
    fi
    python local/scripts/cbi.py ${CBI_LOCATION} ${DATA_PREP_ROOT}
fi


# Humbugdb-Mosquito
if [[ "${DATASETS}" == *"humbugdb"* ]]; then
    log "Processing humbugdb"
    HUMBUGDB_LOCATION="${DATA_PREP_ROOT}/downloads/humbugdb"
    if [ "${BEANS}" == "downloads" ]; then
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
        log "Using data from the provided location: ${BEANS}/humbugdb"
        HUMBUGDB_LOCATION="${BEANS}/humbugdb"
    fi
    python local/scripts/humbugdb.py ${HUMBUGDB_LOCATION} ${DATA_PREP_ROOT}
fi


# Dogs
if [[ "${DATASETS}" == *"dogs"* ]]; then
    log "Processing dogs"
    DOGS_LOCATION="${DATA_PREP_ROOT}/downloads/dogs"
    if [ "${BEANS}" == "downloads" ]; then
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
        log "Using data from the provided location: ${BEANS}/dogs"
        DOGS_LOCATION="${BEANS}/dogs"
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
