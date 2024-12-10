#!/usr/bin/env bash

# Copyright 2020 Aalborg University (Authors: Holger Severin Bovbjerg)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k>]
  optional argument:
    [--min_or_max]: min (Default), max
    [--sample_rate]: 8k (Default), 16k
EOF
)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

cdir=$PWD

# Path to the directory containing MUSAN noise
musan=${MUSAN}

# Path to the directory containing Binaural LibriSpeech
binaural_librispeech="${BINAURAL_LIBRISPEECH}"
binaural_librispeech_subset="horizontal_plane_front_only"

# Subsets to include in data preparation
subsets=("train-clean-100" "train-clean-360" "train-other-500" "dev-clean" "dev-other" "test-clean" "test-other")

min_or_max=max
sample_rate=16k
num_spk=2

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [[ "$min_or_max" != "max" ]] && [[ "$min_or_max" != "min" ]]; then
  echo "Error: min_or_max must be either max or min: ${min_or_max}"
  exit 1
fi
if [[ "$sample_rate" != "16k" ]] && [[ "$sample_rate" != "8k" ]]; then
  echo "Error: sample rate must be either 16k or 8k: ${sample_rate}"
  exit 1
fi

# Simulation parameters
log "Binaural LibriSpeech data preparation"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Downloading BinauralLibriSpeech"
    # Download and extract BinauralLibriSpeech data
    if [ -z "${binaural_librispeech}" ]; then
        if [ -d "${cdir}/downloads/BinauralLibriSpeech" ]; then
          log "Binaural LibriSpeech data already exists in: ${cdir}/downloads/BinauralLibriSpeech."
          log "Skipping download.."
        else
          mkdir -p "${cdir}/downloads/BinauralLibriSpeech"
          log "Downloading data from HuggingFace to '${cdir}/downloads/BinauralLibriSpeech'"
          log "Using Binaural Librispeech version: ${binaural_librispeech_subset}"
          python local/download_binaural_librispeech.py ${binaural_librispeech_subset} "${cdir}/downloads/"
          # Remove HF cache after download is finished.
          rm -rf "${cdir}/downloads/.cache/huggingface/"
          # Move "BinauralLibriSpeech" up one level to "/downloads"
          mv "${cdir}/downloads/data/BinauralLibriSpeech" "${cdir}/downloads"
          # Remove the now-empty "data" folder if desired
          rmdir "${cdir}/downloads/data"
          # Extract data
          log "Extracting data..."
          local/extract_tar_and_delete.sh "${cdir}/downloads/BinauralLibriSpeech"
          log "Downloaded and extracted Binaural Librispeech data"
        fi
    else
        # Copy BinauralLibriSpeech to user directory in case of permission issues.
        log "Binaural LibriSpeech data already exists at ${binaural_librispeech}, copying to '${cdir}/downloads/BinauralLibriSpeech'"
        mkdir -p "${cdir}/downloads/BinauralLibriSpeech"
        log "Using Binaural Librispeech version: ${binaural_librispeech_subset}"
        rsync -r -P "${binaural_librispeech}/${binaural_librispeech_subset}" "${cdir}/downloads/BinauralLibriSpeech"
    fi

    log "stage 0: Downloading MUSAN noise data to '${cdir}/downloads/musan'"
    # Download and extract Musan noise data
    if [ -z "${musan}" ]; then
        mkdir -p ${cdir}/downloads/musan/
        num_wavs=$(find "${cdir}/downloads/musan" -iname "*.wav" | wc -l)
        echo "${num_wavs}"
        if [ "${num_wavs}" = "930" ]; then
            echo "'${cdir}/downloads/musan' already exists. Skipping..."
        else
            musan_noise_url=https://www.openslr.org/resources/17/musan.tar.gz
            wget --continue -O "${cdir}/downloads/musan_noise.tar.gz" ${musan_noise_url}
            tar -zxvf "${cdir}/downloads/musan_noise.tar.gz" -C "${cdir}/downloads/"
            rm "${cdir}/downloads/musan_noise.tar.gz"
            rm -rf "${cdir}/downloads/musan/music/"
            rm -rf "${cdir}/downloads/musan/speech/"
        fi
    else
        log "MUSAN noise data already exists at ${musan}, copying to '${cdir}/downloads/musan'"
        # Copy Musan to user directory in case of permission issues.
        mkdir -p "${cdir}/downloads/musan"
        rsync -r -P "${musan}/noise" "${cdir}/downloads/musan/"

    fi
    # Set paths to downloads
    binaural_librispeech="${cdir}/downloads/BinauralLibriSpeech"
    musan="${cdir}/downloads/musan"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    AUDIO_DIR="${binaural_librispeech}/${binaural_librispeech_subset}"
    NOISE_DIR="${musan}/noise"
    AUDIO_NOISY_DIR="${AUDIO_DIR}_noisy"
    log "stage 1: Simulate noisy data"
    # Check if noisy data already exists and has the same structure
    if [ -d "$AUDIO_NOISY_DIR" ]; then
        log "Noisy data already exists in: $AUDIO_NOISY_DIR"
        log "No further processing is required."
    else
        # Download and extract data or simulate noisy data
        log "Processing audio files from: $AUDIO_DIR"
        log "Using noise files from: $NOISE_DIR"
        log "Saving noisy audio files to: $AUDIO_NOISY_DIR"

        python local/create_noisy_speech.py "$AUDIO_DIR" "$NOISE_DIR" "$AUDIO_NOISY_DIR" ${nj}
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Prepare data in wav.scp, spk2utt , utt2spk and spk1.scp
    log "stage 2: Prepare wav.scp, spk.scp and utt2spk"
    mkdir -p "data/"
    local/prepare_scp.sh ${AUDIO_DIR} ${AUDIO_NOISY_DIR} data
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare data files for each subset"

    for subset in "${subsets[@]}"; do
        mkdir -p "data/${subset}"  # Ensure the directory exists for each subset
        grep -e "${subset}" "data/wav.scp" | sort -u > "data/${subset}/wav.scp" || true  # Continue even if no matches found

        # Copy matching lines from utt2spk based on wav.scp
        awk '{print $1}' "data/${subset}/wav.scp" | grep -Ff - "data/utt2spk" > "data/${subset}/utt2spk" || true

        for f in data/*.scp; do
            [ "$f" = "data/wav.scp" ] || utils/filter_scp.pl "data/${subset}/wav.scp" "$f" > "data/${subset}/$(basename "$f")"
        done

        utils/utt2spk_to_spk2utt.pl data/"${subset}"/utt2spk > data/"${subset}"/spk2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
