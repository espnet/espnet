#!/usr/bin/env bash
#This script is to adjust sampling rate of the wav files and config files.

fs=$1

# fix fbank.conf
sed -e "s/sample-frequency=[0-9]*/sample-frequency=$fs/" ./conf/fbank_16k.conf > ./conf/fbank.conf
# fix pitch.conf
sed -e "s/sample-frequency=[0-9]*/sample-frequency=$fs/" ./conf/pitch_16k.conf > ./conf/pitch.conf
