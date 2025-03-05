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

declare -A DATASETS=(
    ["as2m"]=AUDIOSET
    ["cochlscene"]=COCHLSCENE
    ["epic_sounds"]=EPIC_SOUNDS
    ["fma"]=FMA
    ["inat"]=INAT_SOUNDS
    ["wavcaps"]=WAVCAPS
)

SELECTED_DATASETS=(${@:2})
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    SELECTED_DATASETS=("as2m") # Default: as2m
fi

mkdir -p ${DATA_PREP_ROOT}

# Process selected datasets
for dataset in "${SELECTED_DATASETS[@]}"; do
    VAR_NAME=${DATASETS[$dataset]}
    if [ -z "${!VAR_NAME}" ]; then
        log "Fill the value of ${VAR_NAME} in db.sh"
        exit 1
    fi

    # SCRIPT="local/data_prep_${dataset}.py"
    # if [ -f "$SCRIPT" ]; then
    #     python3 "$SCRIPT" "${!VAR_NAME}" "$DATA_PREP_ROOT"
    # else
    #     log "Script $SCRIPT not found, skipping ${dataset}."
    # fi
done

# copy and append wav.scp files to create training set
mkdir -p ${DATA_PREP_ROOT}/train
mkdir -p ${DATA_PREP_ROOT}/eval
> "${DATA_PREP_ROOT}/train/wav.scp" # clear the file

for x in AudioSet cochlscene epic_sounds fma inat_sounds FreeSound BBC_Sound_Effects SoundBible; do
    if [ ! -d ${DATA_PREP_ROOT}/${x} ]; then
        log "No such directory: ${DATA_PREP_ROOT}/${x}"
        continue
    fi
    cat ${DATA_PREP_ROOT}/${x}/wav.scp >> ${DATA_PREP_ROOT}/train/wav.scp
done
# 7.35M instances total

# make an utt2sk file for all training and eval data
for x in eval train; do
     awk '{print $1, $1}' "${DATA_PREP_ROOT}/${x}/wav.scp" > "${DATA_PREP_ROOT}/${x}/utt2spk"
done

for x in eval train; do
    for f in wav.scp utt2spk; do
        sort ${DATA_PREP_ROOT}/${x}/${f} -o ${DATA_PREP_ROOT}/${x}/${f}
    done
    utils/utt2spk_to_spk2utt.pl ${DATA_PREP_ROOT}/${x}/utt2spk > "${DATA_PREP_ROOT}/${x}/spk2utt"
    utils/validate_data_dir.sh --no-text --no-feats ${DATA_PREP_ROOT}/${x} || exit 1
done

log "Successfully finished. [elapsed=${SECONDS}s]"
