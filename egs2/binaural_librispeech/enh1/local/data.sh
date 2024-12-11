#!/usr/bin/env bash

# Copyright 2024 Aalborg University (Authors: Holger Severin Bovbjerg)
# Apache 2.0
set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=0
stop_stage=100
nj=32
seed=42

. utils/parse_options.sh || exit 1;

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] [--nj <nj>] [--seed <seed>]

  optional argument:
    [--stage]: 0 to 3 (default=0)
    [--stop_stage]: 0 to 100 (default=100)
    [--nj]: number of parallel processes for data simulation (default=4)
    [--seed]: seed for noisy data generation (default=42)
EOF
)

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "${BINAURAL_LIBRISPEECH}" ]; then
    log "Fill the value of 'BINAURAL_LIBRISPEECH' of db.sh"
    exit 1
fi

if [ -z "${MUSAN}" ]; then
    log "Fill the value of 'MUSAN' in db.sh"
    log "(available at http://openslr.org/17/)"
    exit 1
elif [ ! -e "${MUSAN}/noise" ]; then
    log "Please ensure '${MUSAN}/noise' exists"
    exit 1
fi

# Path to current directory
cdir=$PWD

# Path to the directory containing MUSAN noise
musan=${MUSAN}

# Path to the directory containing Binaural LibriSpeech
binaural_librispeech="${BINAURAL_LIBRISPEECH}"
binaural_librispeech_subset="horizontal_plane_front_only"

# Subsets to include in data preparation
subsets=("train-clean-100" "dev-clean" "dev-other" "test-clean" "test-other")
subsets_total_files=39665 # Update number if subset list is changed

if [ $# -ne 0 ]; then
    log "${help_message}"
    exit 1;
fi

# Simulation parameters
log "Binaural LibriSpeech data preparation"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Downloading BinauralLibriSpeech"
    # Download and extract BinauralLibriSpeech data
    if [! -e "${binaural_librispeech}" ]; then
        if [ -d "${cdir}/downloads/BinauralLibriSpeech/${binaural_librispeech_subset}" ]; then
            log "Binaural LibriSpeech data already exists in: ${cdir}/downloads/BinauralLibriSpeech."
            log "Skipping download.."
            # Verify dataset entegrity
            log "Verifying integrity of existing data..."
            n_files=$(find ${cdir}/downloads/BinauralLibriSpeech/${binaural_librispeech_subset} -iname "*.flac" | wc -l)
            expected_files="${subsets_total_files}"
            if [ "$n_files" -eq "$expected_files" ]; then
                log "Check passed: Number of files ($n_files) matches the expected value ($expected_files)."
            else
                log "Check failed: Number of files ($n_files) does not match the expected value ($expected_files)."
                exit 1
            fi
        else
            log "Downloading data from HuggingFace to '${cdir}/downloads/BinauralLibriSpeech'"
            log "Using Binaural Librispeech version: ${binaural_librispeech_subset}"
            mkdir -p "${cdir}/downloads/BinauralLibriSpeech"
            python local/download_binaural_librispeech.py ${binaural_librispeech_subset} "${cdir}/downloads/"
            # Remove HF cache after download is finished.
            rm -rf "${cdir}/downloads/.cache/huggingface/BinauralLibriSpeech/"
            # Move "BinauralLibriSpeech" up one level to "/downloads"
            mv "${cdir}/downloads/data/BinauralLibriSpeech" "${cdir}/downloads"
            # Remove the now-empty "data" folder
            rmdir "${cdir}/downloads/data"
            # Extract data
            log "Extracting data..."
            local/extract_tar_and_delete.sh "${cdir}/downloads/BinauralLibriSpeech"
            log "Downloaded and extracted Binaural Librispeech data"
        fi
    else
        if [ "${binaural_librispeech}" = "downloads" ]; then
            log "Binaural LibriSpeech data already exists at ${cdir}/downloads/BinauralLibriSpeech'."
            # Verify dataset entegrity
            log "Verifying integrity of existing data..."
            n_files=$(find ${cdir}/downloads/BinauralLibriSpeech/${binaural_librispeech_subset} -iname "*.flac" | wc -l)
            expected_files="${subsets_total_files}"
            if [ "$n_files" -eq "$expected_files" ]; then
                log "Check passed: Number of files ($n_files) matches the expected value ($expected_files)."
            else
                log "Check failed: Number of files ($n_files) does not match the expected value ($expected_files)."
                exit 1
            fi
        else
            log "Binaural LibriSpeech data already exists at ${binaural_librispeech}, creating symlink '${cdir}/downloads/BinauralLibriSpeech'"
            # Copy BinauralLibriSpeech to user directory in case of permission issues.
            mkdir -p "${cdir}/downloads/BinauralLibriSpeech"
            log "Using Binaural Librispeech version: ${binaural_librispeech_subset}"
            ln -s "${binaural_librispeech}" "${cdir}/downloads/BinauralLibriSpeech"
            # Verify dataset entegrity
            log "Verifying integrity of existing data..."
            n_files=$(find ${cdir}/downloads/BinauralLibriSpeech/${binaural_librispeech_subset} -iname "*.flac" | wc -l)
            expected_files="${subsets_total_files}"
            if [ "$n_files" -eq "$expected_files" ]; then
                log "Check passed: Number of files ($n_files) matches the expected value ($expected_files)."
            else
                log "Check failed: Number of files ($n_files) does not match the expected value ($expected_files)."
                exit 1
            fi
        fi
    fi

    log "stage 0: Downloading MUSAN noise data to '${cdir}/downloads/musan'"
    # Download and extract Musan noise data
    if [ ! -e "${musan}" ]; then
        mkdir -p ${cdir}/downloads/musan/
        num_wavs=$(find "${cdir}/downloads/musan" -iname "*.wav" | wc -l)
        log "Found ${num_wavs} noise files in ${cdir}/downloads/musan"
        if [ "${num_wavs}" = "930" ]; then
            log "'${cdir}/downloads/musan' already exists. Skipping download..."
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
        # Create
        mkdir -p "${cdir}/downloads/musan"
        ln -s "${musan}" "${cdir}/downloads/musan"

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
    # Check if noisy data already exists
    if [ -d "$AUDIO_NOISY_DIR" ]; then
        log "Noisy data already exists in: $AUDIO_NOISY_DIR"
        log "Verifying integrity of existing noisy data files..."
        n_files=$(find $AUDIO_NOISY_DIR -iname "*.flac" | wc -l)
        expected_files="${subsets_total_files}"
        if [ "$n_files" -eq "$expected_files" ]; then
            log "Check passed: Number of files ($n_files) matches the expected value ($expected_files)."
        else
            log "Check failed: Number of files ($n_files) does not match the expected value ($expected_files). Please check ${AUDIO_NOISY_DIR} for correctness."
            exit 1
        fi
        log "No further processing is required."
    else
        # Simulate noisy data
        log "Processing audio files from: $AUDIO_DIR"
        log "Using noise files from: $NOISE_DIR"
        log "Saving noisy audio files to: $AUDIO_NOISY_DIR"

        python local/create_noisy_speech.py "$AUDIO_DIR" "$NOISE_DIR" "$AUDIO_NOISY_DIR" "$nj" "$seed"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Prepare data in wav.scp, spk2utt , utt2spk and spk1.scp
    log "stage 2: Prepare wav.scp, spk.scp and utt2spk"
    mkdir -p "data/"
    local/prepare_scp.sh "${AUDIO_DIR}" "${AUDIO_NOISY_DIR}" "data" "${subsets_total_files}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare data files for each subset"

    for subset in "${subsets[@]}"; do
        mkdir -p "data/${subset}"  # Ensure the directory exists for each subset
        grep -e "${subset}" "data/wav.scp" | sort -u > "data/${subset}/wav.scp" || true  # Continue even if no matches found

        # Copy matching lines from utt2spk based on wav.scp
        awk '{print $1}' "data/${subset}/wav.scp" | grep -Ff - "data/utt2spk" | sort -u > "data/${subset}/utt2spk" || true

        for f in data/*.scp; do
            [ "$f" = "data/wav.scp" ] || utils/filter_scp.pl "data/${subset}/wav.scp" "$f" > "data/${subset}/$(basename "$f")"
        done

        utils/utt2spk_to_spk2utt.pl data/"${subset}"/utt2spk > data/"${subset}"/spk2utt
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
