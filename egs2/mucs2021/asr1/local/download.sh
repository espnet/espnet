#!/usr/bin/env bash
# This script downloads the three datasets required for the Hinglish ASR project.

# Safety measures for shell scripts.
set -e
set -u
set -o pipefail

# This is the standard ESPnet location for storing raw downloaded data.
# It's a folder named "downloads" inside our current recipe directory.
DOWNLOAD_ROOT="./downloads"

# --- Create the main downloads directory ---
mkdir -p "${DOWNLOAD_ROOT}"
echo "All datasets will be stored in: $(pwd)/${DOWNLOAD_ROOT}"
echo ""

# --- 1. Download MUCS 2021 (Hinglish Code-Switching) ---
echo "--- Downloading MUCS 2021 Dataset ---"
MUCS_URL_TRAIN="https://www.openslr.org/resources/104/Hindi-English_train.tar.gz"
MUCS_URL_TEST="https://www.openslr.org/resources/104/Hindi-English_test.tar.gz"

# This 'if' statement checks if the file already exists before downloading.
if [ ! -f "${DOWNLOAD_ROOT}/Hindi-English_train.tar.gz" ]; then
    echo "Downloading MUCS 2021 training set (7.3 GB)..."
    wget -P "${DOWNLOAD_ROOT}" "${MUCS_URL_TRAIN}"
else
    echo "MUCS 2021 training set already exists. Skipping."
fi

if [ ! -f "${DOWNLOAD_ROOT}/Hindi-English_test.tar.gz" ]; then
    echo "Downloading MUCS 2021 test set (443 MB)..."
    wget -P "${DOWNLOAD_ROOT}" "${MUCS_URL_TEST}"
else
    echo "MUCS 2021 test set already exists. Skipping."
fi
echo "--- MUCS 2021 Download Complete ---"
echo ""

# --- 2. Download LibriSpeech (English Monolingual) ---
echo "--- Downloading LibriSpeech train-clean-100 ---"
LIBRISPEECH_URL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"

if [ ! -f "${DOWNLOAD_ROOT}/train-clean-100.tar.gz" ]; then
    echo "Downloading LibriSpeech train-clean-100 set (6.3 GB)..."
    wget -P "${DOWNLOAD_ROOT}" "${LIBRISPEECH_URL}"
else
    echo "LibriSpeech train-clean-100 already exists. Skipping."
fi
echo "--- LibriSpeech Download Complete ---"
echo ""

# --- 3. Download Gramvaani Hindi (Hindi Monolingual) ---
# A 100-hour labeled dataset of spontaneous telephone speech.
# Source: OpenSLR SLR118
echo "--- Downloading Gramvaani Hindi Dataset ---"
HINDI_URL="https://www.openslr.org/resources/118/GV_Train_100h.tar.gz"
HINDI_DIR="./downloads"

if [ ! -f "${HINDI_DIR}/GV_Train_100h.tar.gz" ]; then
    echo "Downloading Gramvaani Hindi Train set (2.0 GB)..."
    wget -P "${HINDI_DIR}" "${HINDI_URL}"
else
    echo "Gramvaani Hindi Train set already exists. Skipping."
fi
echo "--- Hindi Monolingual Download Complete ---"

echo "All datasets downloaded successfully!"