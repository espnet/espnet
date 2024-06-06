#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# Download audio and transcript files
wget https://indic-asr-public.objectstore.e2enetworks.net/urdu.zip
wget https://indic-asr-public.objectstore.e2enetworks.net/shrutilipi/shrutilipi_fairseq.zip

# Unzip the downloaded files
unzip urdu.zip
unzip shrutilipi_fairseq.zip

# Remove the zip files
rm urdu.zip
rm shrutilipi_fairseq.zip

mkdir -p ${URDU_SHRUTILIPI}
if [ -z "${URDU_SHRUTILIPI}" ]; then
    log "Fill the value of 'URDU_SHRUTILIPI' of db.sh"
    exit 1
fi

python3 data_prep.py
