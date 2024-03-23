#!/usr/bin/env bash

# Author: Atharva Anand Joshi (atharvaa@andrew.cmu.edu)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--parallel <true/false>] [--use_dereverb <true/false>]
  optional argument:
      [--parallel]: true (Default), false
          whether to use parallel execution of the script
      [--use_dereverb]: false (Default), true
          whether to use dereverb or reverb references during training
EOF
)

. ./db.sh
. ./path.sh

### This is a parameter to allow parallel execution of the script.
### It reduces the execution time from ~12 hours to ~1.5 hours
### Number of threads - tr: 50, cv: 13, tt: 8
parallel=true

### This parameter is used to decide whether to use dereverb or reverb references during training.
use_dereverb=false

### The following two parameters are currently fixed for Kinect-WSJ
#min_or_max=min
#sample_rate=16k

output_path=$PWD/data/2speakers_reverb_kinect
tr="tr"
cv="cv"
tt="tt"
pdir=$PWD
spk1_dir="s1"
spk2_dir="s2"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${WSJ0_2MIX}" ]; then
    log "Fill the value of 'WSJ0_2MIX' of db.sh"
    exit 1
fi
if [ ! -e "${CHIME5}" ]; then
    log "Fill the value of 'CHIME5' of db.sh"
    exit 1
fi
if [ ! -e "${DIHARD2}" ]; then
    log "Fill the value of 'DIHARD2' of db.sh"
    exit 1
fi

if [ "${use_dereverb}" == true ]
then
    echo "Using dereverb references"
    spk1_dir="s1_direct"
    spk2_dir="s2_direct"
fi


### Download the scripts for noise extraction and mixture creation
echo "Downloading scripts for wsj_kinect"
repo=https://github.com/sunits/Reverberated_WSJ_2MIX.git
datadir=data
wdir=data/scripts

if [ -d $wdir ]; then
  echo "Replacing $wdir"
  rm -rf ${wdir:?}
fi

for x in ${tr} ${cv} ${tt};
do
  if [ -d $datadir/$x ]; then
  echo "Clearing $x"
  rm -rf ${datadir:?}/${x:?}
  fi
done

mkdir -p $wdir

git clone ${repo} $wdir/Reverberated_WSJ_2MIX-master

cp local/create_corrupted_speech_parallel.sh $wdir/Reverberated_WSJ_2MIX-master/ # Move the modified parallel script to the folder

### Execute the scripts
echo "Running the script with parallel=${parallel}"
cd $wdir/Reverberated_WSJ_2MIX-master
if [ "${parallel}" == true ]
then
  ./create_corrupted_speech_parallel.sh \
  --stage 0 --wsj_data_path ${WSJ0_2MIX} \
  --chime5_wav_base ${CHIME5} --dihard_sad_label_path ${DIHARD2} \
  --dest ${output_path} || exit 1;
else
  chmod 500 create_corrupted_speech.sh
  ./create_corrupted_speech.sh \
  --stage 0 --wsj_data_path ${WSJ0_2MIX} \
  --chime5_wav_base ${CHIME5} --dihard_sad_label_path ${DIHARD2} \
  --dest ${output_path} || exit 1;
fi

### create .scp files for reference audio, noise, direct components and early reflections
cd $pdir
echo "Generating .scp files"
local/wsj_kinect_data_prep.sh ${output_path} $wdir/Reverberated_WSJ_2MIX-master/list

for target_folder in ${tr} ${cv} ${tt};
do
  sed -e "s/\/mix\//\/${spk1_dir}\//g" ./data/$target_folder/wav.scp > ./data/$target_folder/spk1.scp
  sed -e "s/\/mix\//\/${spk2_dir}\//g" ./data/$target_folder/wav.scp > ./data/$target_folder/spk2.scp
  sed -e 's/\/mix\//\/noise\//g' ./data/$target_folder/wav.scp > ./data/$target_folder/noise1.scp
  sed -e 's/\/mix\//\/s1_direct\//g' ./data/$target_folder/wav.scp > ./data/$target_folder/dereverb1.scp
  sed -e 's/\/mix\//\/s2_direct\//g' ./data/$target_folder/wav.scp > ./data/$target_folder/dereverb2.scp
  sed -e 's/\/mix\//\/s1_early\//g' ./data/$target_folder/wav.scp > ./data/$target_folder/spk1_early.scp
  sed -e 's/\/mix\//\/s2_early\//g' ./data/$target_folder/wav.scp > ./data/$target_folder/spk2_early.scp
done
