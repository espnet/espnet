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
DATASETS=${2:-""} #watkins bats cbi humbugdb dogs"}
echo ${DATASETS}

if [ -z "${BEAN}" ]; then
    log "Fill the value of 'BEAN' in db.sh"
    exit 1
fi

# Watkins
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
        log "Using data from the provided location: ${BEANS}/watkins"
        WATKINS_LOCATION="${BEANS}/watkins"
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
        log "Using data from the provided location: ${BEANS}/bats"
        BATS_LOCATION="${BEANS}/bats"
    fi
    python local/scripts/bats.py ${BATS_LOCATION} ${DATA_PREP_ROOT}
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


# #cbi
# local["mkdir"]["-p", "data/cbi/wav"]()
# (
#     local["kaggle"][
#         "competitions", "download", "-p", "data/cbi", "birdsong-recognition"
#     ]
#     & FG
# )
# local["unzip"]["data/cbi/birdsong-recognition.zip", "-d", "data/cbi/"] & FG

# #humbugdb
# (
#     local["git"][
#         "clone", "https://github.com/HumBug-Mosquito/HumBugDB.git", "data/HumBugDB"
#     ]
#     & FG
# )

# for i in [1, 2, 3, 4]:
#     (
#         local["wget"][
#             "-O",
#             f"data/HumBugDB/humbugdb_neurips_2021_{i}.zip",
#             f"https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_{i}.zip?download=1",
#         ]
#         & FG
#     )
#     (
#         local["unzip"][
#             f"data/HumBugDB/humbugdb_neurips_2021_{i}.zip",
#             "-d",
#             "data/HumBugDB/data/audio/",
#         ]
#         & FG
#     )

# #dogs
# local["mkdir"]["-p", "data/dogs/wav"]()
# (
#     local["wget"][
#         "-O",
#         "data/dogs/dog_barks.zip",
#         "https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip",
#     ]
#     & FG
# )
# local["unzip"]["data/dogs/dog_barks.zip", "-d", "data/dogs/"] & FG