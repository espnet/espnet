#!/bin/bash


export PYTHONPATH=../../../:../../TEMPLATE/tse:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh

# Set this to the directory path containing the unpacked LibriSpeech corpus.
#     You will need to download it manually
export LIBRISPEECH=
# Set this to the directory containing the LibriMix dataset.
#     If not set, the script will default to 'data/LibriMix'.
export LIBRIMIX=
