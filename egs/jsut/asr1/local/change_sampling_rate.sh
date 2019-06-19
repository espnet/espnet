#!/bin/bash
#This script is to adjust sampling rate of the wav files and config files.

sampling_rate=$1

# fix fbank.conf
sed -e "s/sample-frequency=[0-9]*/sample-frequency=$sampling_rate/" ./conf/fbank_16k.conf > ./conf/fbank.conf
# fix pitch.conf
sed -e "s/sample-frequency=[0-9]*/sample-frequency=$sampling_rate/" ./conf/pitch_16k.conf > ./conf/pitch.conf
