#!/bin/bash
#This script is to adjust sampling rate of the wav files and config files.

pitch=$1

# fix fbank.conf
sed -i -e "s/sample-frequency=[0-9]*/sample-frequency=$pitch/" ./conf/fbank.conf
# fix pitch.conf
sed -i -e "s/sample-frequency=[0-9]*/sample-frequency=$pitch/" ./conf/pitch.conf
