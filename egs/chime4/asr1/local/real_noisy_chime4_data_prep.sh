#!/usr/bin/env bash
set -e

# Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems.
# - Arnab Ghoshal, 29/05/12

# Modified from the script for CHiME2 baseline
# Shinji Watanabe 02/13/2015
# Modified to use data of six channels
# Szu-Jui Chen 09/29/2017

# Config:
eval_flag=true # make it true when the evaluation data are released

. utils/parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  printf "\nUSAGE: %s <corpus-directory>\n\n" `basename $0`
  echo "The argument should be a the top-level Chime4 directory."
  echo "It is assumed that there will be a 'data' subdirectory"
  echo "within the top-level corpus directory."
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

audio_dir=$(realpath $1/data/audio/16kHz/isolated)
trans_dir=$(realpath $1/data/transcriptions)

echo "extract all channels (CH1-6.wav) for noisy data"

dir=`pwd`/data/local/data
local=`pwd`/local
utils=`pwd`/utils

if ${eval_flag}; then
list_set="tr05_real_noisy dt05_real_noisy et05_real_noisy"
else
list_set="tr05_real_noisy dt05_real_noisy"
fi

cd ${dir}

find ${audio_dir} -name '*CH[1-6].wav' | grep 'tr05_bus_real\|tr05_caf_real\|tr05_ped_real\|tr05_str_real' | sort -u > tr05_real_noisy.flist
find ${audio_dir} -name '*CH[1-6].wav' | grep 'dt05_bus_real\|dt05_caf_real\|dt05_ped_real\|dt05_str_real' | sort -u > dt05_real_noisy.flist
if ${eval_flag}; then
find ${audio_dir} -name '*CH[1-6].wav' | grep 'et05_bus_real\|et05_caf_real\|et05_ped_real\|et05_str_real' | sort -u > et05_real_noisy.flist
fi

# make a dot format from json annotation files
cp ${trans_dir}/tr05_real.dot_all tr05_real.dot
cp ${trans_dir}/dt05_real.dot_all dt05_real.dot
if ${eval_flag}; then
cp ${trans_dir}/et05_real.dot_all et05_real.dot
fi

# make a scp file from file list
for x in ${list_set}; do
    cat ${x}.flist | awk -F'[/]' '{print $NF}'| sed -e 's/\.wav/_REAL/' > ${x}_wav.ids
    paste -d" " ${x}_wav.ids ${x}.flist | sort -k 1 > ${x}_wav.scp
done

#make a transcription from dot
cat tr05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF ".CH1_REAL"}'> tr05_real_noisy.ids
cat tr05_real.dot | sed -e 's/(.*)//' > tr05_real_noisy.txt
paste -d" " tr05_real_noisy.ids tr05_real_noisy.txt | \
awk '{print}{sub(/CH1/, "CH2",$0);print}{sub(/CH2/, "CH3",$0);print}{sub(/CH3/, "CH4",$0);print}{sub(/CH4/, "CH5",$0);print}{sub(/CH5/, "CH6",$0);print}' | \
sort -k 1 > tr05_real_noisy.trans1
cat dt05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF ".CH1_REAL"}'> dt05_real_noisy.ids
cat dt05_real.dot | sed -e 's/(.*)//' > dt05_real_noisy.txt
paste -d" " dt05_real_noisy.ids dt05_real_noisy.txt | \
awk '{print}{sub(/CH1/, "CH2",$0);print}{sub(/CH2/, "CH3",$0);print}{sub(/CH3/, "CH4",$0);print}{sub(/CH4/, "CH5",$0);print}{sub(/CH5/, "CH6",$0);print}' | \
sort -k 1 > dt05_real_noisy.trans1
if ${eval_flag}; then
cat et05_real.dot | sed -e 's/(\(.*\))/\1/' | awk '{print $NF ".CH1_REAL"}'> et05_real_noisy.ids
cat et05_real.dot | sed -e 's/(.*)//' > et05_real_noisy.txt
paste -d" " et05_real_noisy.ids et05_real_noisy.txt | \
awk '{print}{sub(/CH1/, "CH2",$0);print}{sub(/CH2/, "CH3",$0);print}{sub(/CH3/, "CH4",$0);print}{sub(/CH4/, "CH5",$0);print}{sub(/CH5/, "CH6",$0);print}' | \
sort -k 1 > et05_real_noisy.trans1
fi

# Do some basic normalization steps.  At this point we don't remove OOVs--
# that will be done inside the training scripts, as we'd like to make the
# data-preparation stage independent of the specific lexicon used.
noiseword="<NOISE>";
for x in ${list_set};do
  cat ${x}.trans1 | ${local}/normalize_transcript.pl ${noiseword} \
    | sort > ${x}.txt || exit 1;
done

# Make the utt2spk and spk2utt files.
for x in ${list_set}; do
  cat ${x}_wav.scp | awk -F'_' '{print $1}' > ${x}.spk
  cat ${x}_wav.scp | awk '{print $1}' > ${x}.utt
  paste -d" " ${x}.utt ${x}.spk > ${x}.utt2spk
  cat ${x}.utt2spk | ${utils}/utt2spk_to_spk2utt.pl > ${x}.spk2utt || exit 1;
done

# copying data to data/...
for x in ${list_set}; do
  mkdir -p ../../${x}
  cp ${x}_wav.scp ../../${x}/wav.scp || exit 1;
  cp ${x}.txt     ../../${x}/text    || exit 1;
  cp ${x}.spk2utt ../../${x}/spk2utt || exit 1;
  cp ${x}.utt2spk ../../${x}/utt2spk || exit 1;
done

echo "Data preparation succeeded"
