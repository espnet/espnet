#!/bin/bash

# Download audio and transcript files
wget https://indic-asr-public.objectstore.e2enetworks.net/urdu.zip
wget https://indic-asr-public.objectstore.e2enetworks.net/shrutilipi/shrutilipi_fairseq.zip

# Unzip the downloaded files
unzip urdu.zip
unzip shrutilipi_fairseq.zip

# Remove the zip files
rm urdu.zip
rm shrutilipi_fairseq.zip

mkdir data

python3 data_prep.py