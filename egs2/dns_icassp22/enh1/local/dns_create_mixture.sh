#!/usr/bin/env bash

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dns> <dns_wav> <total_hours> <nj>"
  echo " where <dns> is dns downloaded data directory,"
  echo " <dns_wav> is wav generation directory."
  echo " <total_hours> is the duration (in hours) of synthetic noisy data to generate."
  echo " <nj> is the number of jobs created to synthesize noisy data."
  exit 1;
fi

dns=$1
dns_wav=$2
total_hours=$3
nj=$4

rm -r data/ 2>/dev/null || true
mkdir -p data/

rm -rf ${dns_wav} 2>/dev/null || true
mkdir -p ${dns_wav}


# modify path in the original noisyspeech_synthesizer.cfg
configure=${dns}/noisyspeech_synthesizer.cfg
train_cfg=data/noisyspeech_synthesizer.cfg


if [ ! -f ${configure} ]; then
  echo -e "Please check configurtion ${configure} exist"
  exit 1;
fi

# input datas
noise_dir=${dns}/datasets_fullband/noise_fullband
speech_dir=${dns}/datasets_fullband/clean_fullband/read_speech

# additional clean datas (not used by default)
use_singing_data=0
use_mandarin_data=0
use_emotion_data=0
clean_singing=${dns}/datasets_fullband/clean_fullband/singing_voice
clean_emotion=${dns}/datasets_fullband/clean_fullband/emotional_speech
clean_mandarin=${dns}/datasets_fullband/clean_fullband/mandarin_speech

# outputs
noisy_wav=${dns_wav}/noisy
clean_wav=${dns_wav}/clean
noise_wav=${dns_wav}/noise
log_dir=log

# modify the input paths for "\" separated paths
sed -e "/^noisy_destination/s#.*#noisy_destination:${noisy_wav}#g"  \
    -e "/^clean_destination/s#.*#clean_destination:${clean_wav}#g"  \
    -e "/^noise_destination/s#.*#noise_destination:${noise_wav}#g"  \
    -e "/^noise_dir/s#.*#noise_dir:${noise_dir}#g"  \
    -e "/^speech_dir/s#.*#speech_dir:${speech_dir}#g"  \
    -e "/^use_singing_data/s#.*#use_singing_data:${use_singing_data}#g"  \
    -e "/^use_emotion_data/s#.*#use_emotion_data:${use_emotion_data}#g"  \
    -e "/^use_mandarin_data/s#.*#use_mandarin_data:${use_mandarin_data}#g"  \
    -e "/^clean_singing/s#.*#clean_singing:${clean_singing}#g"  \
    -e "/^clean_emotion/s#.*#clean_emotion:${clean_emotion}#g"  \
    -e "/^clean_mandarin/s#.*#clean_mandarin:${clean_mandarin}#g"  \
    -e "/^total_hours/s#.*#total_hours:${total_hours}#g"  \
    -e "/^log_dir/s#.*#log_dir:${log_dir}#g" ${configure} \
  > ${train_cfg}

echo "nj:${nj}" >> ${train_cfg}


mix_script=local/noisyspeech_synthesizer.py

if [ ! -f ${configure} -a -f ${mix_script} ]; then
  echo -e "Please check configurtion ${configure} and mix_script ${mix_script} exist"
  exit 1;
fi

echo "Creating Mixtures for Training and Validation Data."
python ${mix_script} --cfg ${PWD}/${train_cfg} >/dev/null || exit 1;