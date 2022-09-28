#!/usr/bin/bash

# Adapted from DNS-Challenge official repo file: https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-4.sh
# Original License: CC-BY 4.0 International: https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE
# Retrieved on Sep. 7th, 2022, by Shih-Lun Wu (summer7sean@gmail.com) 

# ***** Datasets for ICASSP 2022 DNS Challenge 4 - Main (Real-Time) Track *****

# NOTE: Before downloading, make sure you have enough space
# on your local storage!

# In all, you will need about 1TB to store the UNPACKED data.
# Archived, the same data takes about 550GB total.

# Please comment out the files you don't need before launching
# the script.

# NOTE: By default, the script *DOES NOT* DOWNLOAD ANY FILES!
# Please scroll down and edit this script to pick the
# downloading method that works best for you.

# -------------------------------------------------------------
# The directory structure of the unpacked data is:

# datasets_fullband 892G
# +-- dev_testset 1.7G
# +-- impulse_responses 5.9G
# +-- noise_fullband 58G
# \-- clean_fullband 827G
#     +-- emotional_speech 2.4G
#     +-- french_speech 62G
#     +-- german_speech 319G
#     +-- italian_speech 42G
#     +-- read_speech 299G
#     +-- russian_speech 12G
#     +-- spanish_speech 65G
#     +-- vctk_wav48_silence_trimmed 27G
#     \-- VocalSet_48kHz_mono 974M

# NOTE: for storage space considerations, by default, only English read speech data will be downloaded (Shih-Lun)
BLOB_NAMES=(
    # clean_fullband/datasets_fullband.clean_fullband.VocalSet_48kHz_mono_000_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.emotional_speech_000_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_002_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_003_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_005_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_006_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_007_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_008_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.german_speech_000_0.00_3.47.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_001_3.47_3.64.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_002_3.64_3.74.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_003_3.74_3.81.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_004_3.81_3.86.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_005_3.86_3.91.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_006_3.91_3.96.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_007_3.96_4.00.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_008_4.00_4.04.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_009_4.04_4.08.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_010_4.08_4.12.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_011_4.12_4.16.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_012_4.16_4.21.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_013_4.21_4.26.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_014_4.26_4.33.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_015_4.33_4.43.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_016_4.43_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_017_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_018_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_019_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_020_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_021_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_022_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_023_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_024_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_025_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_026_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_027_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_028_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_029_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_030_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_031_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_032_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_033_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_034_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_035_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_036_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_037_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_038_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_039_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_040_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_041_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.german_speech_042_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_000_0.00_3.98.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_001_3.98_4.21.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_002_4.21_4.40.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_003_4.40_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.italian_speech_005_NA_NA.tar.bz2

    clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_001_3.75_3.88.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_002_3.88_3.96.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_003_3.96_4.02.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_004_4.02_4.06.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_005_4.06_4.10.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_006_4.10_4.13.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_007_4.13_4.16.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_008_4.16_4.19.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_009_4.19_4.21.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_010_4.21_4.24.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_011_4.24_4.26.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_012_4.26_4.29.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_013_4.29_4.31.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_014_4.31_4.33.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_015_4.33_4.35.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_016_4.35_4.38.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_017_4.38_4.40.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_018_4.40_4.42.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_019_4.42_4.45.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_020_4.45_4.48.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_021_4.48_4.52.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_022_4.52_4.57.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_023_4.57_4.67.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_024_4.67_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_025_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_026_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_027_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_028_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_029_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_030_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_031_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_032_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_033_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_034_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_035_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_036_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_037_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_038_NA_NA.tar.bz2
    clean_fullband/datasets_fullband.clean_fullband.read_speech_039_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.russian_speech_000_0.00_4.31.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.russian_speech_001_4.31_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_000_0.00_4.09.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_001_4.09_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_002_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_003_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_005_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_006_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_007_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.spanish_speech_008_NA_NA.tar.bz2

    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_000.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_001.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_002.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_003.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.vctk_wav48_silence_trimmed_004.tar.bz2

    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_002.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_003.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_004.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_005.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.audioset_006.tar.bz2

    noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2
    noise_fullband/datasets_fullband.noise_fullband.freesound_001.tar.bz2

    # datasets_fullband.dev_testset_000.tar.bz2
    # datasets_fullband.impulse_responses_000.tar.bz2
)

###############################################################

AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
PATH_PREFIX=$1

RUN_DIR=$(pwd)

mkdir -p $PATH_PREFIX
mkdir -p $PATH_PREFIX/{clean_fullband,noise_fullband}

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"

    # DRY RUN: print HTTP response and Content-Length
    # WITHOUT downloading the files
    # curl -s -I "$URL" | head -n 2

    wget "$URL" -O ${PATH_PREFIX}/$BLOB -c
    if [[ $(dirname ${PATH_PREFIX}/$BLOB) != $PATH_PREFIX ]]; then
      cd $(dirname ${PATH_PREFIX}/$BLOB)
      cd ../
      mv $BLOB ./
    fi
    tar -xjf $(basename $BLOB)
done

# delete folders created only to prevent path errors
rmdir $PATH_PREFIX/clean_fullband
rmdir $PATH_PREFIX/noise_fullband

# get noisy wav synthesizer config file
cd ${PATH_PREFIX}
wget https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/noisyspeech_synthesizer.cfg -O noisyspeech_synthesizer.cfg

cd $RUN_DIR/local/
wget https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/audiolib.py -O audiolib.py
wget https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/utils.py -O utils.py

cd ../